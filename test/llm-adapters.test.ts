/**
 * Network-free adapter tests. Each adapter is handed a STUB SDK client that
 * records the request it received and returns a canned response, so we assert
 * both directions of the translation (neutral → wire, wire → neutral) with no
 * key and no network.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { OpenAIChatModel } from "../src/ai/llm/openai.js";
import { AnthropicChatModel } from "../src/ai/llm/anthropic.js";
import type { ChatRequest } from "../src/ai/llm/types.js";

// --- OpenAI adapter ---------------------------------------------------------

function fakeOpenAI(resp: unknown) {
  const box: { params?: any } = {};
  const client = {
    chat: { completions: { create: async (params: any) => ((box.params = params), resp) } },
  } as unknown as OpenAI;
  return { client, box };
}

function openAiResp(over: Partial<any> = {}): any {
  return {
    choices: [{ message: { content: "hello", tool_calls: [] }, finish_reason: "stop" }],
    usage: { prompt_tokens: 100, completion_tokens: 20, prompt_tokens_details: { cached_tokens: 30 } },
    ...over,
  };
}

const REJECTED_OBJECT_TOOL_CHOICE = "Invalid tool_choice type: 'object'. Supported string values: none, auto, required";

test("openai: plain text — system/user map to strings, usage is uncached-only", async () => {
  const { client, box } = fakeOpenAI(openAiResp());
  const m = new OpenAIChatModel({ apiKey: "x", model: "gpt-x", client });
  const res = await m.create({
    messages: [
      { role: "system", content: "sys" },
      { role: "user", content: "hi" },
    ],
    temperature: 0,
  });
  assert.equal(box.params.model, "gpt-x");
  assert.equal(box.params.messages[0].content, "sys"); // plain string, no cache parts
  assert.equal(box.params.temperature, 0); // forwarded on OpenAI-compatible
  assert.equal(box.params.tools, undefined);
  assert.equal(res.text, "hello");
  assert.deepEqual(res.usage, { input: 70, output: 20, cacheRead: 30, cacheCreate: 0 });
});

test("openai: cacheBreakpoint turns content into a cache_control part", async () => {
  const { client, box } = fakeOpenAI(openAiResp());
  const m = new OpenAIChatModel({ apiKey: "x", model: "gpt-x", client });
  await m.create({ messages: [{ role: "user", content: "hi", cacheBreakpoint: true }] });
  const part = box.params.messages[0].content[0];
  assert.equal(part.type, "text");
  assert.deepEqual(part.cache_control, { type: "ephemeral" });
});

test("openai: forced tool — args come back PARSED", async () => {
  const resp = openAiResp({
    choices: [
      {
        message: {
          content: null,
          tool_calls: [{ id: "c1", type: "function", function: { name: "record_graph", arguments: '{"nodes":[1,2]}' } }],
        },
        finish_reason: "tool_calls",
      },
    ],
  });
  const { client, box } = fakeOpenAI(resp);
  const m = new OpenAIChatModel({ apiKey: "x", model: "gpt-x", client });
  const res = await m.create({
    messages: [{ role: "user", content: "go" }],
    tools: [{ name: "record_graph", description: "d", parameters: { type: "object" } }],
    responseFormat: { kind: "tool", name: "record_graph" },
  });
  assert.deepEqual(box.params.tool_choice, { type: "function", function: { name: "record_graph" } });
  assert.equal(res.toolCalls.length, 1);
  assert.deepEqual(res.toolCalls[0].args, { nodes: [1, 2] });
});

test("openai: json mode routes through a synthetic forced tool and returns JSON text", async () => {
  const resp = openAiResp({
    choices: [
      {
        message: {
          content: null,
          tool_calls: [{ id: "c1", type: "function", function: { name: "emit_json", arguments: '{"correct":true}' } }],
        },
        finish_reason: "tool_calls",
      },
    ],
  });
  const { client, box } = fakeOpenAI(resp);
  const m = new OpenAIChatModel({ apiKey: "x", model: "gpt-x", client });
  const res = await m.create({ messages: [{ role: "user", content: "grade" }], responseFormat: { kind: "json" } });
  assert.equal(box.params.tool_choice.function.name, "emit_json");
  assert.equal(res.text, '{"correct":true}');
  assert.equal(res.toolCalls.length, 0); // synthetic tool hidden
});

test("openai: assistant providerRaw replays verbatim", async () => {
  const { client, box } = fakeOpenAI(openAiResp());
  const m = new OpenAIChatModel({ apiKey: "x", model: "gpt-x", client });
  const raw = { role: "assistant", content: "verbatim", extra: 1 };
  await m.create({
    messages: [
      { role: "user", content: "hi" },
      { role: "assistant", content: "reconstructed", providerRaw: { provider: "openai", raw } },
    ],
  });
  assert.deepEqual(box.params.messages[1], raw);
});

test("openai: unparseable tool arguments stay undefined instead of becoming {}", async () => {
  const { client } = fakeOpenAI(
    openAiResp({
      choices: [
        {
          message: {
            content: "",
            tool_calls: [
              { id: "1", type: "function", function: { name: "record_graph", arguments: '{"nodes":[{"name":' } },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
    }),
  );
  const m = new OpenAIChatModel({ apiKey: "x", model: "gpt-x", client });
  const res = await m.create({
    messages: [{ role: "user", content: "q" }],
    responseFormat: { kind: "tool", name: "record_graph" },
  });
  assert.equal(res.toolCalls[0].args, undefined);
});

test("openai: json mode keeps the model's text when the synthetic call's args did not parse", async () => {
  const { client } = fakeOpenAI(
    openAiResp({
      choices: [
        {
          message: {
            content: "not JSON, sorry",
            tool_calls: [{ id: "j1", type: "function", function: { name: "emit_json", arguments: '{"correct":' } }],
          },
          finish_reason: "tool_calls",
        },
      ],
    }),
  );
  const m = new OpenAIChatModel({ apiKey: "x", model: "gpt-x", client });
  const res = await m.create({ messages: [{ role: "user", content: "grade" }], responseFormat: { kind: "json" } });
  assert.equal(res.text, "not JSON, sorry");
  assert.equal(res.toolCalls.length, 0);
});

test("openai: retries with tool_choice \"required\" when the server rejects the object form (single tool)", async () => {
  const calls: any[] = [];
  const resp = openAiResp({
    choices: [
      {
        message: {
          content: null,
          tool_calls: [{ id: "c1", type: "function", function: { name: "emit_json", arguments: '{"correct":true}' } }],
        },
        finish_reason: "tool_calls",
      },
    ],
  });
  const client = {
    chat: {
      completions: {
        create: async (params: any) => {
          calls.push(params);
          if (calls.length === 1) {
            throw new OpenAI.APIError(400, { message: REJECTED_OBJECT_TOOL_CHOICE }, REJECTED_OBJECT_TOOL_CHOICE, new Headers());
          }
          return resp;
        },
      },
    },
  } as unknown as OpenAI;
  const m = new OpenAIChatModel({ apiKey: "x", model: "local-model", client });
  const res = await m.create({ messages: [{ role: "user", content: "grade" }], responseFormat: { kind: "json" } });

  assert.equal(calls.length, 2); // first attempt (object form) + retry (string form)
  assert.deepEqual(calls[0].tool_choice, { type: "function", function: { name: "emit_json" } });
  assert.equal(calls[1].tool_choice, "required");
  assert.equal(res.text, '{"correct":true}');
});

test("openai: does NOT paper over a rejected object tool_choice when multiple tools are offered", async () => {
  let callCount = 0;
  const client = {
    chat: {
      completions: {
        create: async () => {
          callCount++;
          throw new OpenAI.APIError(400, { message: REJECTED_OBJECT_TOOL_CHOICE }, REJECTED_OBJECT_TOOL_CHOICE, new Headers());
        },
      },
    },
  } as unknown as OpenAI;
  const m = new OpenAIChatModel({ apiKey: "x", model: "local-model", client });
  await assert.rejects(
    () =>
      m.create({
        messages: [{ role: "user", content: "go" }],
        tools: [
          { name: "a", description: "d", parameters: { type: "object" } },
          { name: "b", description: "d", parameters: { type: "object" } },
        ],
        responseFormat: { kind: "tool", name: "a" },
      }),
    OpenAI.APIError,
  );
  assert.equal(callCount, 1); // no ambiguous retry — the caller asked for "a" specifically
});

// --- Anthropic adapter ------------------------------------------------------

function fakeAnthropic(resp: unknown) {
  const box: { params?: any } = {};
  const client = {
    messages: { create: async (params: any) => ((box.params = params), resp) },
  } as unknown as Anthropic;
  return { client, box };
}

function anthropicResp(over: Partial<any> = {}): any {
  return {
    content: [{ type: "text", text: "hi there" }],
    stop_reason: "end_turn",
    usage: { input_tokens: 70, output_tokens: 20, cache_read_input_tokens: 30, cache_creation_input_tokens: 5 },
    ...over,
  };
}

test("anthropic: system is hoisted, temperature dropped, max_tokens defaulted", async () => {
  const { client, box } = fakeAnthropic(anthropicResp());
  const m = new AnthropicChatModel({ apiKey: "x", model: "claude-x", client });
  const res = await m.create({
    messages: [
      { role: "system", content: "sys" },
      { role: "user", content: "hi" },
    ],
    temperature: 0, // must NOT be forwarded
  });
  assert.equal(box.params.system[0].text, "sys");
  assert.equal(box.params.messages.length, 1);
  assert.equal(box.params.messages[0].role, "user");
  assert.equal(box.params.temperature, undefined);
  assert.equal(box.params.max_tokens, 4096);
  assert.deepEqual(res.usage, { input: 70, output: 20, cacheRead: 30, cacheCreate: 5 });
});

test("anthropic: consecutive tool results coalesce into ONE user turn", async () => {
  const { client, box } = fakeAnthropic(anthropicResp());
  const m = new AnthropicChatModel({ apiKey: "x", model: "claude-x", client });
  const req: ChatRequest = {
    messages: [
      { role: "user", content: "q" },
      { role: "assistant", content: "", toolCalls: [{ id: "a", name: "t", args: {} }, { id: "b", name: "t", args: {} }] },
      { role: "tool", toolCallId: "a", content: "ra" },
      { role: "tool", toolCallId: "b", content: "rb" },
    ],
  };
  await m.create(req);
  const msgs = box.params.messages;
  const lastUser = msgs[msgs.length - 1];
  assert.equal(lastUser.role, "user");
  assert.equal(lastUser.content.length, 2); // both tool_result blocks in one turn
  assert.equal(lastUser.content[0].tool_use_id, "a");
  assert.equal(lastUser.content[1].tool_use_id, "b");
});

test("anthropic: tool_use input is an object (no JSON.parse round-trip)", async () => {
  const resp = anthropicResp({
    content: [{ type: "tool_use", id: "u1", name: "record_graph", input: { nodes: [1] } }],
    stop_reason: "tool_use",
  });
  const { client } = fakeAnthropic(resp);
  const m = new AnthropicChatModel({ apiKey: "x", model: "claude-x", client });
  const res = await m.create({
    messages: [{ role: "user", content: "go" }],
    tools: [{ name: "record_graph", description: "d", parameters: { type: "object" } }],
    responseFormat: { kind: "tool", name: "record_graph" },
  });
  assert.deepEqual(res.toolCalls[0].args, { nodes: [1] });
});

test("anthropic: reconstructed assistant tool_use carries the object input", async () => {
  const { client, box } = fakeAnthropic(anthropicResp());
  const m = new AnthropicChatModel({ apiKey: "x", model: "claude-x", client });
  await m.create({
    messages: [
      { role: "user", content: "q" },
      { role: "assistant", content: "", toolCalls: [{ id: "u1", name: "t", args: { k: 1 } }] },
      { role: "tool", toolCallId: "u1", content: "res" },
    ],
  });
  const asst = box.params.messages[1];
  assert.equal(asst.content[0].type, "tool_use");
  assert.deepEqual(asst.content[0].input, { k: 1 });
});

test("anthropic: json mode forces emit_json and returns serialized text", async () => {
  const resp = anthropicResp({
    content: [{ type: "tool_use", id: "j1", name: "emit_json", input: { correct: true } }],
    stop_reason: "tool_use",
  });
  const { client, box } = fakeAnthropic(resp);
  const m = new AnthropicChatModel({ apiKey: "x", model: "claude-x", client });
  const res = await m.create({ messages: [{ role: "user", content: "grade" }], responseFormat: { kind: "json" } });
  assert.deepEqual(box.params.tool_choice, { type: "tool", name: "emit_json" });
  assert.equal(res.text, '{"correct":true}');
  assert.equal(res.toolCalls.length, 0);
});

const THINKING_TOOL_CHOICE = "Thinking mode does not support this tool_choice";

/** Stub client whose first call is refused the way a thinking-by-default endpoint refuses it. */
function refusingAnthropic(resp: unknown) {
  const calls: Anthropic.MessageCreateParamsNonStreaming[] = [];
  const client = {
    messages: {
      create: async (params: Anthropic.MessageCreateParamsNonStreaming) => {
        calls.push(params);
        if (calls.length === 1) {
          throw new Anthropic.APIError(400, { message: THINKING_TOOL_CHOICE }, THINKING_TOOL_CHOICE, new Headers());
        }
        return resp;
      },
    },
  } as unknown as Anthropic; // stub SDK client, same seam as fakeAnthropic above
  return { client, calls };
}

test("anthropic: retries with thinking disabled when the model refuses a forced tool_choice", async () => {
  const resp = anthropicResp({
    content: [{ type: "tool_use", id: "s1", name: "record_symbols", input: { symbols: [] } }],
    stop_reason: "tool_use",
  });
  const { client, calls } = refusingAnthropic(resp);
  const m = new AnthropicChatModel({ apiKey: "x", model: "deepseek-flash", client });
  const res = await m.create({
    messages: [{ role: "user", content: "describe" }],
    tools: [{ name: "record_symbols", description: "d", parameters: { type: "object" } }],
    responseFormat: { kind: "tool", name: "record_symbols" },
  });
  assert.equal(calls.length, 2);
  assert.equal(calls[0].thinking, undefined);
  assert.deepEqual(calls[1].thinking, { type: "disabled" });
  assert.deepEqual(calls[1].tool_choice, { type: "tool", name: "record_symbols" }); // the caller's tool survived
  assert.equal(res.toolCalls.length, 1);
});

test("anthropic: does NOT disable thinking for a request that forces no tool", async () => {
  const { client, calls } = refusingAnthropic(anthropicResp());
  const m = new AnthropicChatModel({ apiKey: "x", model: "deepseek-flash", client });
  await assert.rejects(() => m.create({ messages: [{ role: "user", content: "hi" }] }));
  assert.equal(calls.length, 1); // no rewrite of a request the caller did not force
});

test("anthropic: the thinking retry happens before streaming a big budget", async () => {
  const resp = anthropicResp({
    content: [{ type: "tool_use", id: "s1", name: "record_symbols", input: { symbols: [] } }],
    stop_reason: "tool_use",
  });
  const calls: string[] = [];
  const streamed: Anthropic.MessageCreateParamsNonStreaming[] = [];
  const client = {
    messages: {
      create: async () => {
        calls.push("create");
        if (calls.length === 1) {
          throw new Anthropic.APIError(400, { message: THINKING_TOOL_CHOICE }, THINKING_TOOL_CHOICE, new Headers());
        }
        throw new Anthropic.AnthropicError("Streaming is required for operations that may take longer than 10 minutes.");
      },
      stream: (params: Anthropic.MessageCreateParamsNonStreaming) => {
        calls.push("stream");
        streamed.push(params);
        return { finalMessage: async () => resp };
      },
    },
  } as unknown as Anthropic; // stub SDK client, same seam as fakeAnthropic above
  const m = new AnthropicChatModel({ apiKey: "x", model: "deepseek-flash", client });
  const res = await m.create({
    messages: [{ role: "user", content: "describe" }],
    maxTokens: 32768,
    tools: [{ name: "record_symbols", description: "d", parameters: { type: "object" } }],
    responseFormat: { kind: "tool", name: "record_symbols" },
  });
  assert.deepEqual(calls, ["create", "create", "stream"]); // never stream the thinking-mode refusal
  assert.deepEqual(streamed[0].thinking, { type: "disabled" });
  assert.deepEqual(streamed[0].tool_choice, { type: "tool", name: "record_symbols" });
  assert.equal(res.toolCalls.length, 1);
});

test("anthropic: a budget above the SDK's non-streaming cap is streamed", async () => {
  const resp = anthropicResp({
    content: [{ type: "tool_use", id: "s1", name: "record_symbols", input: { symbols: [] } }],
    stop_reason: "tool_use",
  });
  const calls = { create: 0, stream: 0 };
  const client = {
    messages: {
      create: async () => {
        calls.create++;
        // Verbatim shape of the SDK's refusal (see its _calculateNonstreamingTimeout).
        throw new Anthropic.AnthropicError(
          "Streaming is required for operations that may take longer than 10 minutes. See https://github.com/anthropics/anthropic-sdk-typescript#long-requests for more details",
        );
      },
      stream: () => {
        calls.stream++;
        return { finalMessage: async () => resp };
      },
    },
  } as unknown as Anthropic;
  const m = new AnthropicChatModel({ apiKey: "x", model: "deepseek-flash", client });
  const res = await m.create({
    messages: [{ role: "user", content: "describe" }],
    maxTokens: 32768,
    tools: [{ name: "record_symbols", description: "d", parameters: { type: "object" } }],
    responseFormat: { kind: "tool", name: "record_symbols" },
  });
  assert.deepEqual(calls, { create: 1, stream: 1 });
  assert.equal(res.toolCalls.length, 1);
});
