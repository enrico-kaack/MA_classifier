from io import BytesIO
from tokenize import tokenize, tok_name
import more_itertools
from tqdm.autonotebook import tqdm
import logging


def process_general_data(data, vocab, window_size=20, step_size=3, problem_type="RETURN_NULL", encode_type=True):
    x = []
    y = []
    for d in tqdm(data):
        problem_line_numbers = [l['line_number'] for l in d['problems'] if l['type'] == problem_type]
        tokens= tokenize(BytesIO(d['src'].encode('utf-8')).readline)
        token_list = []
        for index, (token_type, token_string, start, end, source_line) in enumerate(tokens):
            token_list.append((index, token_type, token_string, start, end, source_line))

        windowed_tokens = list(more_itertools.windowed(token_list, n=window_size, step=step_size))

        y_single = _extract_labels_for_window(windowed_tokens, token_list, problem_line_numbers)
        x_single = _encode_input_vector(windowed_tokens, vocab, encode_type)
        if (len(x_single) != len(y_single)):
            logging.warning(f"x and y size not same for file{d['file_path']}")
            continue
        x.extend(x_single)
        y.extend(y_single)

    return x,y

def decode_vector(tokens, reverse_vocab, encode_type=True):
    decoded = []
    for index, token in enumerate(tokens):
        if index % 2 == 0 and encode_type:
            # token type
            if token == 257: # unknown
                decoded.append(">UNKNOWN>")
            else:
                decoded.append(tok_name[token])
        else:
            # token value, reverse dict search
            token_value = reverse_vocab.get(token, None)
            if token_value is None:
                token_value = "<UNKNOWN>"
            decoded.append(token_value)
    return decoded

def _extract_labels_for_window(windowed_tokens, token_list, problem_line_numbers):
    #for every problem line, get start and end token (for that line)
    problems_with_start_end = []
    for problem_line in problem_line_numbers:
        tokens_index_of_that_line = [token for token in token_list if token[3][0] == problem_line]
        start_token_index = min(tokens_index_of_that_line)
        end_token_index = max(tokens_index_of_that_line)
        problems_with_start_end.append({"start": start_token_index, "end": end_token_index, "line_number": problem_line})

    # check if start.row contains problem_line_numbers and if line is completely in this sample
    y = []
    for window in windowed_tokens:
        first_token = [t for t in window if t is not None][0]
        last_token = [t for t in window if t is not None][-1]
        y_single = 0
        for problem in problems_with_start_end:
            if problem["start"][0] >= first_token[0] and problem["end"][0] <= last_token[0]:
                y_single = 1
        y.append(y_single)
    return y

def _encode(token, vocab, encode_type=True):
    if encode_type:
        if token is None:
            return 257, len(vocab)+1
        token_type_encoded = token[1]
        token_value_encoding = vocab.get(token[2].lower(), len(vocab)+1)
        return token_type_encoded, token_value_encoding
    else:
        if token is None:
            return len(vocab)+1
        token_value_encoding = vocab.get(token[2].lower(), len(vocab)+1)
        return token_value_encoding


def _encode_input_vector(windowed_tokens, vocab, encode_type=True):
    x = []
    for window in windowed_tokens:
        window_x = []
        for token in window:
            if encode_type:
                type_encoded, value_encoded = _encode(token, vocab, encode_type)
                window_x.extend([type_encoded, value_encoded])
            else:
                value_encoded = _encode(token, vocab, encode_type)
                window_x.extend([value_encoded])
        x.append(window_x)
    return x