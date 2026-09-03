/**
 * Construct the normalized sessions used by built-in Transformers.js models.
 */
export async function constructSessions(names, options, cache_sessions = undefined) {
    if (!options.inferenceProvider?.constructSessions) {
        throw new Error('The selected inference provider does not support built-in model sessions.');
    }
    return options.inferenceProvider.constructSessions(names, options, cache_sessions);
}

/**
 * Run a normalized built-in model session.
 */
export async function sessionRun(session, inputs) {
    return session.run(inputs);
}
