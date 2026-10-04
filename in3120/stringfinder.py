# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods
# pylint: disable=protected-access

from dataclasses import dataclass
from typing import Iterator, Any, List
from .analyzer import Analyzer
from .trie import Trie


class StringFinder:
    """
    Given a trie encoding a dictionary of strings, efficiently finds the subset of strings in the dictionary
    that are also present in a given text buffer. I.e., in a sense computes the "intersection" or "overlap"
    between the dictionary and the text buffer.

    Uses a trie-walk algorithm similar to the Aho-Corasick algorithm with some simplifications (we ignore the
    part about failure transitions) and some minor NLP extensions. The running time of this algorithm is in
    practice virtually insensitive to the size of the dictionary, and linear in the length of the buffer we
    are searching in.

    The analyzer we use when scanning the input buffer is assumed to be the same as the one that was used
    when adding strings to the trie.
    """

    @dataclass
    class State:
        """
        A currently explored state, as the scan proceeds.
        """
        node: Trie  # The current position in the trie, after having consumed zero or more characters.
        begin: int  # The index into the original buffer where the state was "born".
        match: str  # The symbols consumed so far to get to the current state.

    @dataclass
    class Result:
        """
        An individual result of the scan, as reported back to the client.
        """
        match: str        # The matching dictionary entry.
        meta: None | Any  # Optional mata data associated with the match, if present in the dictionary.
        surface: str      # The part of the input buffer that triggered the match, space-normalized.
        begin: int        # The index into the original buffer where the surface form starts.
        end: int          # The index into the original buffer where the surface form ends.

    def __init__(self, trie: Trie, analyzer: Analyzer):
        self._trie = trie          # The set of strings we want to detect in the scanned buffer.
        self._analyzer = analyzer  # The same that was used when the trie was built.

    def scan(self, buffer: str) -> Iterator[Result]:
        """
        Scans the given buffer once and finds all dictionary entries in the trie that are also present in the
        buffer. We only consider matches that begin and end on token boundaries.

        In a serious application we'd add more lookup/evaluation features, e.g., support for prefix matching,
        support for leftmost-longest matching (instead of reporting all matches), and more.
        """

        start_state = self._trie # trie root
        terms_list = list(self._analyzer.terms(buffer))  #[(token, (0, 5)), ...]
        partial_matches: List[StringFinder.State] = []
        previous_term_end = None

        for index, (term, (begin, end)) in enumerate(terms_list):
            matches_for_next_term: List[StringFinder.State] = []

            # Add a space between normal words
            if previous_term_end is None or previous_term_end == begin:
                separator = ""
            else:
                separator = " "

            input_to_consume = separator + term

            # Keep a match only if it can continue
            if index < len(terms_list) - 1:  # if this is no the final term
                next_item = terms_list[index + 1]
                next_term = next_item[0]
                next_begin = next_item[1][0]

                # check if the current term ends where the next term begin
                if end == next_begin: # if no space between the two
                    next_input = next_term[0]
                else:
                    next_input = " " + next_term[0] # if separated with a space, next_input starts with a space
            else:
                next_input = None # the last term has no next input

            # extend matches that has already picked up
            for partial_match in partial_matches:
                node_after_term = partial_match.node.consume(input_to_consume) # try adding input to this partial match
                if node_after_term is None: # skip this partial match if None
                    continue

                matched_text = partial_match.match + input_to_consume

                # node is final: found a dictionary entry
                if node_after_term.is_final():
                    surface = " ".join(buffer[partial_match.begin:end].split())

                    yield self.Result(matched_text,node_after_term.get_meta(),surface,partial_match.begin,end)

                # retain the match only when it can consume the actual next input
                if next_input is not None and node_after_term.consume(next_input) is not None:
                    matches_for_next_term.append(self.State(node_after_term,partial_match.begin,matched_text))

            # start a new match at the current term
            node_after_term = start_state.consume(term)

            if node_after_term is not None:
                # The current term is a dictionary entry
                if node_after_term.is_final():
                    surface = " ".join(buffer[begin:end].split())
                    yield self.Result(term,node_after_term.get_meta(),surface,begin,end)

                # keep it if it can become a longer dictionary entry
                if next_input is not None and node_after_term.consume(next_input) is not None:
                    matches_for_next_term.append(self.State(node_after_term,begin,term))

            # alive matches to be used for processing next term
            partial_matches = matches_for_next_term
            previous_term_end = end
