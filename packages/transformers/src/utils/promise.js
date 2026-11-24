/**
 * Await `primary`, and keep `background` from becoming an unhandled rejection
 * if `primary` fails first. `background` is still awaited afterwards, so its
 * own failure still surfaces when `primary` succeeds.
 * @template T
 * @template U
 * @param {Promise<T>} primary
 * @param {Promise<U>} background
 * @returns {Promise<[T, U]>}
 */
export async function awaitWithBackground(primary, background) {
    background.catch(() => {});
    const primaryResult = await primary;
    const backgroundResult = await background;
    return [primaryResult, backgroundResult];
}
