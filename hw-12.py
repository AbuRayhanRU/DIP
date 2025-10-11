import heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, namedtuple
from PIL import Image

class HuffNode(namedtuple("HuffNode", ["symbol", "count", "left", "right"])):
    def __lt__(self, other):
        return self.count < other.count


def count_frequencies(data):
    """Return frequency dictionary of symbols."""
    return Counter(data)


def create_huffman_tree(freq_dict):
    """Build Huffman tree using a min-heap."""
    heap = []
    for sym, freq in freq_dict.items():
        heapq.heappush(heap, HuffNode(sym, freq, None, None))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffNode(None, left.count + right.count, left, right)
        heapq.heappush(heap, merged)

    return heap[0]


def make_codebook(root):
    """Generate Huffman codes (forward and reverse mappings)."""
    codebook, reverse_map = {}, {}

    def traverse(node, code=""):
        if node is None:
            return
        if node.symbol is not None:
            codebook[node.symbol] = code
            reverse_map[code] = node.symbol
            return
        traverse(node.left, code + "0")
        traverse(node.right, code + "1")

    traverse(root)
    return codebook, reverse_map


def huffman_encode(data, codebook):
    """Encode input data using Huffman codes."""
    return "".join(codebook[val] for val in data)


def pad_bitstring(bitstring):
    """Pad bitstring to make length a multiple of 8."""
    padding_len = 8 - len(bitstring) % 8
    bitstring += "0" * padding_len
    pad_info = f"{padding_len:08b}"
    return pad_info + bitstring


def bitstring_to_bytes(bitstring):
    """Convert bitstring to bytearray."""
    return bytearray(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))


def remove_padding(padded_bits):
    """Remove padding info and return pure encoded bits."""
    pad_info = padded_bits[:8]
    pad_len = int(pad_info, 2)
    return padded_bits[8:][:-pad_len]


def huffman_decode(encoded_bits, reverse_map):
    """Decode bitstring using reverse mapping."""
    temp_code = ""
    decoded = []
    for bit in encoded_bits:
        temp_code += bit
        if temp_code in reverse_map:
            decoded.append(reverse_map[temp_code])
            temp_code = ""
    return decoded


def compress_image(image_array):
    """Compress image array using Huffman coding."""
    flat = image_array.flatten()
    freq_dict = count_frequencies(flat)
    tree_root = create_huffman_tree(freq_dict)
    codebook, reverse_map = make_codebook(tree_root)

    encoded = huffman_encode(flat, codebook)
    padded = pad_bitstring(encoded)
    compressed_bytes = bitstring_to_bytes(padded)

    original_bits = len(flat) * 8
    compressed_bits = len(padded)
    return compressed_bytes, codebook, reverse_map, original_bits, compressed_bits


def decompress_image(comp_bytes, reverse_map, shape):
    """Decompress Huffman-coded image bytes."""
    bit_data = "".join(bin(byte)[2:].rjust(8, "0") for byte in comp_bytes)
    encoded = remove_padding(bit_data)
    decoded = huffman_decode(encoded, reverse_map)
    return np.array(decoded, dtype=np.uint8).reshape(shape)


IMG_PATH = "/home/rayhan/Downloads/image7.jpg"
image_gray = np.array(Image.open(IMG_PATH).convert("L"))

# Compress
compressed_bytes, codebook, reverse_map, orig_bits, comp_bits = compress_image(image_gray)

# Decompress
reconstructed = decompress_image(compressed_bytes, reverse_map, image_gray.shape)

# Compression Ratio
ratio = orig_bits / comp_bits
print(f"Original Size: {orig_bits} bits")
print(f"Compressed Size: {comp_bits} bits")
print(f"Compression Ratio: {ratio:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image_gray, cmap="gray")
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(reconstructed, cmap="gray")
axs[1].set_title("Decompressed Image")
axs[1].axis("off")

plt.tight_layout()
#plt.savefig("images/output/huffman_refactored_result.png", dpi=300)
plt.show()
