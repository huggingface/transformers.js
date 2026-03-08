function normalizeMergedWordText(text) {
    return String(text ?? '')
        .normalize('NFKC')
        .toLowerCase()
        .replace(/^[("'“‘\[{]+/g, '')
        .replace(/[.,!?;:)"'”’\]}]+$/g, '')
        .trim();
}

export function dedupeMergedWords(words) {
    /** @type {typeof words} */
    const merged = [];
    for (const word of words) {
        const prev = merged.at(-1);
        if (
            prev &&
            normalizeMergedWordText(prev.text) === normalizeMergedWordText(word.text) &&
            word.startTime < prev.endTime
        ) {
            const prevDuration = prev.endTime - prev.startTime;
            const nextDuration = word.endTime - word.startTime;
            if (nextDuration > prevDuration) {
                merged[merged.length - 1] = word;
            }
            continue;
        }
        merged.push(word);
    }
    return merged;
}
