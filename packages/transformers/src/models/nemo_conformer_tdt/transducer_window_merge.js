function normalizeMergedWordText(text) {
    return String(text ?? '')
        .normalize('NFKC')
        .toLowerCase()
        .replace(/^[("'“‘\[{]+/g, '')
        .replace(/[.,!?;:)"'”’\]}]+$/g, '')
        .trim();
}

function normalizeRawMergedWordText(text) {
    return String(text ?? '')
        .normalize('NFKC')
        .toLowerCase()
        .trim();
}

export function dedupeMergedWords(words) {
    /** @type {typeof words} */
    const merged = [];
    for (const word of words) {
        const prev = merged.at(-1);
        const prevText = normalizeMergedWordText(prev?.text);
        const wordText = normalizeMergedWordText(word.text);
        if (
            prev &&
            prevText === wordText &&
            (
                prevText.length > 0 ||
                normalizeRawMergedWordText(prev.text) === normalizeRawMergedWordText(word.text)
            ) &&
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
