const HASH_ALGORITHM = "SHA-256";

class CrossOriginStorage {
  static isAvailable = () =>
    typeof navigator !== "undefined" && "crossOriginStorage" in navigator;

  match = async (request) => {
    const hashValue = await this._getFileHash(request);
    if (!hashValue) {
      return undefined;
    }
    const hash = { algorithm: HASH_ALGORITHM, value: hashValue };
    try {
      // @ts-expect-error
      const [handle] = await navigator.crossOriginStorage.requestFileHandles([
        hash,
      ]);
      const blob = await handle.getFile();
      return new Response(blob);
    } catch (err) {
      return undefined;
    }
  };
  put = async (request, response) => {
    const blob = await response.blob();
    const hash = await this._getBlobHash(blob);
    // @ts-expect-error
    const [handle] = await navigator.crossOriginStorage.requestFileHandles(
      [hash],
      { create: true },
    );
    const writableStream = await handle.createWritable();
    await writableStream.write(blob);
    await writableStream.close();
  };

  // Gets the SHA-256 hash for large resources as per
  // https://huggingface.co/docs/hub/en/storage-backends#xet.
  _getFileHash = async (url) => {
    if (/\/resolve\/main\/onnx\//.test(url)) {
      const rawUrl = url.replace(/\/resolve\//, "/raw/");
      const text = await fetch(rawUrl).then((response) => response.text());
      if (!text.includes("oid sha256:")) {
        return null;
      }
      return text.replace(/.*?\n^oid sha256:(\w+)\n.*?$/gm, "$1") || null;
    }
    return null;
  };

  _getBlobHash = async (blob) => {
    const hashAlgorithmIdentifier = "SHA-256";

    // Get the contents of the blob as binary data contained in an ArrayBuffer.
    const arrayBuffer = await blob.arrayBuffer();

    // Hash the arrayBuffer using SHA-256.
    const hashBuffer = await crypto.subtle.digest(
      hashAlgorithmIdentifier,
      arrayBuffer,
    );

    // Convert the ArrayBuffer to a hex string.
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");

    return {
      algorithm: hashAlgorithmIdentifier,
      value: hashHex,
    };
  };
}

export default CrossOriginStorage;
