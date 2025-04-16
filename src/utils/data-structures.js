
/**
 * @file Custom data structures.
 * 
 * These are only used internally, meaning an end-user shouldn't
 * need to access anything here.
 * 
 * @module utils/data-structures
 */


/**
 * Efficient Heap-based Implementation of a Priority Queue.
 * It uses an array-based binary heap, where the root is at index `0`, and the
 * children of node `i` are located at indices `2i + 1` and `2i + 2`, respectively.
 * 
 * Adapted from the following sources:
 * - https://stackoverflow.com/a/42919752/13989043 (original)
 * - https://github.com/belladoreai/llama-tokenizer-js (minor improvements)
 */
export class PriorityQueue {

    /**
     * Create a new PriorityQueue.
     * @param {function(any, any): boolean} comparator Comparator function to determine priority. Defaults to a MaxHeap.
     */
    constructor(comparator = (a, b) => a > b, maxSize = Infinity) {
        this._heap = [];
        this._comparator = comparator;
        this._maxSize = maxSize;
    }

    /**
     * The size of the queue
     */
    get size() {
        return this._heap.length;
    }

    /**
     * Check if the queue is empty.
     * @returns {boolean} `true` if the queue is empty, `false` otherwise.
     */
    isEmpty() {
        return this.size === 0;
    }

    /**
     * Return the element with the highest priority in the queue.
     * @returns {any} The highest priority element in the queue.
     */
    peek() {
        return this._heap[0];
    }

    /**
     * Add one or more elements to the queue.
     * @param  {...any} values The values to push into the queue.
     * @returns {number} The new size of the queue.
     */
    push(...values) {
        return this.extend(values);
    }

    /**
     * Add multiple elements to the queue.
     * @param {any[]} values The values to push into the queue.
     * @returns {number} The new size of the queue.
     */
    extend(values) {
        for (const value of values) {
            if (this.size < this._maxSize) {
                this._heap.push(value);
                this._siftUp();
            } else {
                // Get index of value with the lowest priority
                const smallest = this._smallest();

                // If the new value has higher priority than the smallest value in the heap
                // then replace the smallest value with the new value and update the heap
                if (this._comparator(value, this._heap[smallest])) {
                    this._heap[smallest] = value;
                    this._siftUpFrom(smallest);
                }
            }
        }
        return this.size;
    }

    /**
     * Remove and return the element with the highest priority in the queue.
     * @returns {any} The element with the highest priority in the queue.
     */
    pop() {
        const poppedValue = this.peek();
        const bottom = this.size - 1;
        if (bottom > 0) {
            this._swap(0, bottom);
        }
        this._heap.pop();
        this._siftDown();
        return poppedValue;
    }

    /**
     * Replace the element with the highest priority in the queue with a new value.
     * @param {*} value The new value.
     * @returns {*} The replaced value.
     */
    replace(value) {
        const replacedValue = this.peek();
        this._heap[0] = value;
        this._siftDown();
        return replacedValue;
    }

    /**
     * Compute the index for the parent of the node at index `i`.
     * @param {number} i The index of the node to get the parent of.
     * @returns {number} The index of the parent node.
     * @private
     */
    _parent(i) {
        return ((i + 1) >>> 1) - 1;
    }

    /**
     * Compute the index for the left child of the node at index `i`.
     * @param {number} i The index of the node to get the left child of.
     * @returns {number} The index of the left child.
     * @private
     */
    _left(i) {
        return (i << 1) + 1;
    }

    /**
     * Compute the index for the right child of the node at index `i`.
     * @param {number} i The index of the node to get the right child of.
     * @returns {number} The index of the right child.
     * @private
     */
    _right(i) {
        return (i + 1) << 1;
    }

    /**
     * Check if the element at index `i` is greater than the element at index `j`.
     * @param {number} i The index of the first element to compare.
     * @param {number} j The index of the second element to compare.
     * @returns {boolean} `true` if the element at index `i` is greater than the element at index `j`, `false` otherwise.
     * @private
     */
    _greater(i, j) {
        return this._comparator(this._heap[i], this._heap[j]);
    }

    /**
     * Swap the elements at indices `i` and `j`.
     * @param {number} i The index of the first element to swap.
     * @param {number} j The index of the second element to swap.
     * @private
     */
    _swap(i, j) {
        const temp = this._heap[i];
        this._heap[i] = this._heap[j];
        this._heap[j] = temp;
    }

    /**
     * Maintain the heap property by updating positions in the heap,
     * starting at the last element and moving up the heap.
     * @private
     */
    _siftUp() {
        this._siftUpFrom(this.size - 1);
    }

    /**
     * Helper function to sift up from a given node.
     * @param {number} node The index of the node to start sifting up from.
     */
    _siftUpFrom(node) {
        while (node > 0 && this._greater(node, this._parent(node))) {
            this._swap(node, this._parent(node));
            node = this._parent(node);
        }
    }

    /**
     * Maintain the heap property by updating positions in the heap,
     * starting at the first element and moving down the heap.
     * @private
     */
    _siftDown() {
        let node = 0;
        while (
            (this._left(node) < this.size && this._greater(this._left(node), node)) ||
            (this._right(node) < this.size && this._greater(this._right(node), node))
        ) {
            const maxChild = (this._right(node) < this.size && this._greater(this._right(node), this._left(node)))
                ? this._right(node)
                : this._left(node);
            this._swap(node, maxChild);
            node = maxChild;
        }
    }

    /**
     * Get the index of the smallest element in the heap. Since we use an array-based heap,
     * the index can be computed without needing to traverse the heap.
     * @private
     */
    _smallest() {
        return (2 ** (Math.floor(Math.log2(this.size))) - 1);
    }
}

/**
 * A trie structure to efficiently store and search for strings.
 */
export class CharTrie {
    constructor() {
        this.root = CharTrieNode.default();
    }

    /**
     * Adds one or more `texts` to the trie.
     * @param {string[]} texts The strings to add to the trie.
     */
    extend(texts) {
        for (const text of texts) {
            this.push(text);
        }
    }

    /**
     * Adds text to the trie.
     * @param {string} text The string to add to the trie.
     */
    push(text) {
        let node = this.root;
        for (const ch of text) {
            let child = node.children.get(ch);
            if (child === undefined) {
                child = CharTrieNode.default();
                node.children.set(ch, child);
            }
            node = child;
        }
        node.isLeaf = true;
    }

    /**
     * Searches the trie for all strings with a common prefix of `text`.
     * @param {string} text The common prefix to search for.
     * @yields {string} Each string in the trie that has `text` as a prefix.
     */
    *commonPrefixSearch(text) {
        let node = this.root;
        if (node === undefined) return;

        let prefix = "";
        for (const ch of text) {
            prefix += ch;
            node = node.children.get(ch);
            if (node === undefined) return;
            if (node.isLeaf) {
                yield prefix;
            }
        }
    }
}

/**
 * Represents a node in a character trie.
 */
class CharTrieNode {
    /**
     * Create a new CharTrieNode.
     * @param {boolean} isLeaf Whether the node is a leaf node or not.
     * @param {Map<string, CharTrieNode>} children A map containing the node's children, where the key is a character and the value is a `CharTrieNode`.
     */
    constructor(isLeaf, children) {
        this.isLeaf = isLeaf;
        this.children = children;
    }

    /**
     * Returns a new `CharTrieNode` instance with default values.
     * @returns {CharTrieNode} A new `CharTrieNode` instance with `isLeaf` set to `false` and an empty `children` map.
     */
    static default() {
        return new CharTrieNode(false, new Map());
    }
}

/**
 * A lattice data structure to be used for tokenization.
 */
export class TokenLattice {
    /**
     * Creates a new TokenLattice instance.
     *
     * @param {string} sentence The input sentence to be tokenized.
     * @param {number} bosTokenId The beginning-of-sequence token ID.
     * @param {number} eosTokenId The end-of-sequence token ID.
     */
    constructor(sentence, bosTokenId, eosTokenId) {
        this.chars = Array.from(sentence);
        this.len = this.chars.length;
        this.bosTokenId = bosTokenId;
        this.eosTokenId = eosTokenId;
        this.nodes = [];
        this.beginNodes = Array.from({ length: this.len + 1 }, () => []);
        this.endNodes = Array.from({ length: this.len + 1 }, () => []);

        const bos = new TokenLatticeNode(this.bosTokenId, 0, 0, 0, 0.0);
        const eos = new TokenLatticeNode(this.eosTokenId, 1, this.len, 0, 0.0);
        this.nodes.push(bos.clone());
        this.nodes.push(eos.clone());
        this.beginNodes[this.len].push(eos);
        this.endNodes[0].push(bos);
    }

    /**
     * Inserts a new token node into the token lattice.
     *
     * @param {number} pos The starting position of the token.
     * @param {number} length The length of the token.
     * @param {number} score The score of the token.
     * @param {number} tokenId The token ID of the token.
     */
    insert(pos, length, score, tokenId) {
        const nodeId = this.nodes.length;
        const node = new TokenLatticeNode(tokenId, nodeId, pos, length, score);
        this.beginNodes[pos].push(node);
        this.endNodes[pos + length].push(node);
        this.nodes.push(node);
    }

    /**
     * Implements the Viterbi algorithm to compute the most likely sequence of tokens.
     *
     * @returns {TokenLatticeNode[]} The most likely sequence of tokens.
     */
    viterbi() {
        const len = this.len;
        let pos = 0;
        while (pos <= len) {
            if (this.beginNodes[pos].length == 0) {
                return [];
            }
            for (let rnode of this.beginNodes[pos]) {
                rnode.prev = null;
                let bestScore = 0.0;
                let bestNode = null;
                for (let lnode of this.endNodes[pos]) {
                    const score = lnode.backtraceScore + rnode.score;
                    if (bestNode === null || score > bestScore) {
                        bestNode = lnode.clone();
                        bestScore = score;
                    }
                }

                if (bestNode !== null) {
                    rnode.prev = bestNode;
                    rnode.backtraceScore = bestScore;
                } else {
                    return [];
                }
            }
            ++pos;
        }

        const results = [];
        const root = this.beginNodes[len][0];
        const prev = root.prev;
        if (prev === null) {
            return [];
        }

        let node = prev.clone();
        while (node.prev !== null) {
            results.push(node.clone());
            const n = node.clone();
            node = n.prev.clone();
        }

        results.reverse();
        return results;
    }

    /**
     * @param {TokenLatticeNode} node
     * @returns {string} The array of nodes representing the most likely sequence of tokens.
     */
    piece(node) {
        return this.chars.slice(node.pos, node.pos + node.length).join('');
    }

    /**
     * @returns {string[]} The most likely sequence of tokens.
     */
    tokens() {
        const nodes = this.viterbi();
        return nodes.map(x => this.piece(x));
    }

    /**
     * @returns {number[]} The most likely sequence of token ids.
     */
    tokenIds() {
        const nodes = this.viterbi();
        return nodes.map(x => x.tokenId);
    }
}
class TokenLatticeNode {
    /**
     * Represents a node in a token lattice for a given sentence.
     * @param {number} tokenId The ID of the token associated with this node.
     * @param {number} nodeId The ID of this node.
     * @param {number} pos The starting position of the token in the sentence.
     * @param {number} length The length of the token.
     * @param {number} score The score associated with the token.
     */
    constructor(tokenId, nodeId, pos, length, score) {
        this.tokenId = tokenId;
        this.nodeId = nodeId;
        this.pos = pos;
        this.length = length;
        this.score = score;
        this.prev = null;
        this.backtraceScore = 0.0;
    }

    /**
     * Returns a clone of this node.
     * @returns {TokenLatticeNode} A clone of this node.
     */
    clone() {
        const n = new TokenLatticeNode(this.tokenId, this.nodeId, this.pos, this.length, this.score);
        n.prev = this.prev;
        n.backtraceScore = this.backtraceScore;
        return n;
    }
}

/**
 * A data structure which uses a trie to split a string into tokens based on a dictionary.
 * It can also use a regular expression to preprocess the input text before splitting.
 * 
 * NOTE: To ensure multi-byte characters are handled correctly, we operate at byte-level instead of character-level.
 */
export class DictionarySplitter {
    /**
     * @param {string[]} dictionary The dictionary of words to use for splitting.
     */
    constructor(dictionary) {
        this.trie = this._buildTrie(dictionary);
    }

    /**
     * Builds a trie from the given dictionary.
     * @param {string[]} dictionary The dictionary of words to build the trie from.
     * @returns {Object} The root node of the trie.
     * @private
     */
    _buildTrie(dictionary) {
        const trie = Object.create(null);
        for (const word of dictionary) {
            let node = trie;
            for (let i = 0; i < word.length; ++i) {
                node = (node[word[i]] ??= Object.create(null));
            }
            node.end = word;
        }
        return trie;
    }

    /**
     * Splits the input text into tokens based on the dictionary.
     * @param {string} text The input text to split.
     * @returns {string[]} An array of tokens.
     */
    split(text) {
        const result = [];
        const n = text.length;
        let start = 0;
        let i = 0;

        while (i < n) {
            let node = this.trie;
            let match = null;
            let j = i;

            while (j < n && (node = node[text[j]])) {
                if (node.end) {
                    // Always keep the last (i.e., longest) match.
                    match = node.end;
                }
                ++j;
            }

            if (match) {
                if (i > start) {
                    result.push(text.slice(start, i));
                }
                result.push(match);
                i += match.length;
                start = i;
            } else {
                ++i;
            }
        }
        if (start < n) {
            result.push(text.slice(start));
        }
        return result;
    }
}

/**
* A simple Least Recently Used (LRU) cache implementation in JavaScript.
* This cache stores key-value pairs and evicts the least recently used item
* when the capacity is exceeded.
*/
export class LRUCache {
    /**
     * Creates an LRUCache instance.
     * @param {number} capacity The maximum number of items the cache can hold.
     */
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map();
    }

    /**
     * Retrieves the value associated with the given key and marks the key as recently used.
     * @param {any} key The key to retrieve.
     * @returns {any} The value associated with the key, or undefined if the key does not exist.
     */
    get(key) {
        if (!this.cache.has(key)) return undefined;
        const value = this.cache.get(key);
        this.cache.delete(key);
        this.cache.set(key, value);
        return value;
    }

    /**
     * Inserts or updates the key-value pair in the cache.
     * If the key already exists, it is updated and marked as recently used.
     * If the cache exceeds its capacity, the least recently used item is evicted.
     * @param {any} key The key to add or update.
     * @param {any} value The value to associate with the key.
     */
    put(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        }
        this.cache.set(key, value);
        if (this.cache.size > this.capacity) {
            this.cache.delete(this.cache.keys().next().value);
        }
    }

    /**
     * Clears the cache.
     */
    clear() {
        this.cache.clear();
    }
}
