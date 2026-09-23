/**
 * Extraction that survives a dead grammar runtime.
 *
 * Every WASM grammar in graft (the breadth tier in generic.ts, the container tier
 * in container.ts) shares ONE web-tree-sitter module instance, and that instance is
 * a wasm32 module: its heap tops out at 2048 MiB and its C allocator `abort()`s when
 * a request cannot be met. Emscripten's abort is not a per-call error — it latches
 * `ABORT = true` on the module and every later call into it throws `Aborted(…)`, so
 * a single exhausted heap used to turn into a "parse failed" line for every file
 * parsed after it, each memoized in the extract cache as that file's own failure.
 *
 * So an abort is treated as what it is: a process-level condition to recover from.
 * `extractResilient` restarts the runtime (generic.ts `resetGrammarRuntime`, a fresh
 * module instance — an aborted one cannot be revived in place) and re-parses the same
 * file once. The retry is what keeps a good file from being recorded as unparseable
 * because of the file that went before it.
 *
 * This lives above both tiers rather than in generic.ts because generic.ts cannot
 * import container.ts (container.ts imports generic.ts).
 */
import { extractFile, type ExtractResult, type Language } from "./extract.js";
import {
  extractGeneric,
  resetGrammarRuntime,
  warmGenericGrammars,
} from "./generic.js";
import {
  extractContainer,
  resetContainerGrammars,
  warmContainerGrammars,
  type ContainerLang,
} from "./container.js";

/** The tier a single file is extracted by — exactly one of the three is set, in
 *  the same precedence `buildGraph`/`checkGraph` use: depth, then container, then
 *  breadth. */
export interface ExtractTier {
  depth?: Language;
  generic?: string;
  container?: ContainerLang;
}

/** How many runtime restarts one build/check run may spend. Each restart re-warms
 *  every grammar, so this is a bound on the cost of a pathological repo, not a
 *  per-file limit. */
export interface RestartBudget {
  used: number;
  max: number;
}

/** Emscripten's abort message is `Aborted(<what>). Build with -sASSERTIONS…`, and
 *  `extractGeneric` prefixes it with `<lang> grammar threw: `. Deliberately narrow:
 *  other wasm traps ("memory access out of bounds", #139) do NOT kill the module
 *  and must keep their per-file-error treatment. */
export function isRuntimeAbort(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err);
  return message.includes("Aborted(");
}

/** Restart the shared runtime and re-warm both WASM tiers for the repo being built.
 *  The names come from the caller because the file list does: only it knows which
 *  grammars this repo needs. */
export async function restartWasmRuntime(
  genericNames: Iterable<string>,
  containerNames: Iterable<string>,
): Promise<void> {
  await resetGrammarRuntime();
  resetContainerGrammars();
  await warmGenericGrammars(genericNames);
  await warmContainerGrammars(containerNames);
}

/** Extract one file by its tier, recovering once from a dead grammar runtime.
 *
 *  1. extract → return
 *  2. a non-abort error, or an exhausted budget → rethrow
 *  3. spend a restart, re-warm, retry the same file ONCE; a second throw propagates
 *
 *  A `restart()` that itself fails rethrows the original abort: the caller records
 *  that one file and stops, rather than replacing a diagnosable error with a loader
 *  error. The container tier never throws, so it never takes this path.
 */
export async function extractResilient(
  rel: string,
  source: string,
  tier: ExtractTier,
  opts: { restart: () => Promise<void>; budget: RestartBudget },
): Promise<ExtractResult> {
  const attempt = (): ExtractResult => {
    if (tier.depth) return extractFile(rel, source, tier.depth);
    if (tier.container) return extractContainer(rel, source, tier.container);
    if (tier.generic) return extractGeneric(rel, source, tier.generic);
    // Reachable only if a file reaches the loop that no tier claims. It used to be
    // a `generic!.name` TypeError; keep it an error (recorded per file, never
    // silently cached as a clean empty parse) but say what happened.
    throw new Error("no extractor claims this file");
  };
  try {
    return attempt();
  } catch (err) {
    if (!isRuntimeAbort(err) || opts.budget.used >= opts.budget.max) throw err;
    opts.budget.used++;
    try {
      await opts.restart();
    } catch {
      throw err; // the runtime is gone for good — report the parse, not the restart
    }
    return attempt();
  }
}
