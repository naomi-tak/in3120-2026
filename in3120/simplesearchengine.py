# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from select import KQ_NOTE_EXEC
from typing import Iterator, List, Tuple

from . import SimpleAnalyzer
from .document import Document
from .sieve import Sieve
from .ranker import Ranker
from .corpus import Corpus
from .posting import Posting
from .invertedindex import InvertedIndex



class SimpleSearchEngine:
    """
    Realizes a simple query evaluator that efficiently performs N-of-M matching over an inverted index.
    I.e., if the query contains M unique query terms, each document in the result set should contain at
    least N of these M terms. For example, 2-of-3 matching over the query 'orange apple banana' would be
    logically equivalent to the following predicate:

       (orange AND apple) OR (orange AND banana) OR (apple AND banana)
       
    Note that N-of-M matching can be viewed as a type of "soft AND" evaluation, where the degree of match
    can be smoothly controlled to mimic either an OR evaluation (1-of-M), or an AND evaluation (M-of-M),
    or something in between.

    The evaluator uses the client-supplied ratio T = N/M as a parameter as specified by the client on a
    per query basis. For example, for the query 'john paul george ringo' we have M = 4 and a specified
    threshold of T = 0.75 would imply that at least 3 of the 4 query terms have to be present in a matching
    document.
    """

    @dataclass
    class Options:
        """
        Query-time options. Controls lookup behavior.
        """
        match_threshold: float = 0.5  # The required ratio N/M, as described above.
        hit_count: int = 10           # The maximum number of results to return to the client.

    @dataclass
    class Result:
        """
        An individual lookup result, as reported back to the client.
        """
        score: float        # The matching document's relevance score, as determined by the ranker.
        document: Document  # The document with the matching content.

    @dataclass
    class Cursor:
        """
        State while DAAT-traversing a query term's posting list.
        """
        term: str                    # The query term this cursor is for.
        multiplicity: int            # How many times the query term appears in the query.
        current: Posting | None      # The current posting for the query term, or None if we've exhausted the posting list.
        postings: Iterator[Posting]  # The posting list for the query term.

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex):
        self._corpus = corpus
        self._inverted_index = inverted_index

    def _alive(self, cursors: List[Cursor]) -> List[int]:
        """
        Returns the subset of cursors (or, rather, their indices) that are still "alive",
        i.e., the subset of cursors having posting lists that have not yet been exhausted.
        """
        alive_cursors = []
        for i, cursor in enumerate(cursors):
            if cursor.current is not None:
                alive_cursors.append(i)
        return alive_cursors
    
    def _advance(self, cursors: List[Cursor], subset: List[int]) -> None:
        """"
        Advances the given subset of cursors.
        """
        for i in subset:
            cursor = cursors[i]
            cursor.current = next(cursor.postings, None)

        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def _frontier(self, cursors: List[Cursor], subset: List[int]) -> Tuple[int, List[int]]:
        """
        Among the given subset of cursors, identifies the frontier document identifier
        and the cursors positioned at it.

        The frontier is defined as the subset of non-exhausted posting lists that reference
        the smallest document identifier. Since posting lists are sorted in ascending order,
        the frontier represents the "leftmost" cursors when scanning from left to right.
        """
        # frontier_doc_id
        i = subset[0]
        frontier_doc_id = cursors[i].current.document_id

        for i in subset[1:]:
            doc_id = cursors[i].current.document_id
            if doc_id < frontier_doc_id:
                frontier_doc_id = doc_id

        # cursors positioned at frontier_doc_id
        frontier_cursors_indices = []
        for i in subset:
            if cursors[i].current.document_id == frontier_doc_id:
                frontier_cursors_indices.append(i)

        return frontier_doc_id, frontier_cursors_indices

        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")



    def evaluate(self, query: str, ranker: Ranker, options: Options | None = None) -> Iterator[Result]:
        """
        Evaluates the given query, doing N-out-of-M ranked retrieval. I.e., for a supplied query having M
        unique terms, a document is considered to be a match if it contains at least N <= M of those terms.

        The matching documents, if any, are ranked by the supplied ranker, and only the "best" matches are yielded
        back to the client.
        """
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

        # normalize query
        analyzer = SimpleAnalyzer()
        query_normalized = analyzer.join(query).split()
        multiplicity = Counter(query_normalized)
        m = len(set(query_normalized))     # num. unique query term
        t = options.match_threshold
        n = max(1, min(m, int(t * m)))      # num. terms to be matched


        # initialize Cursor obj for each query term
        cursors = []
        for term, count in multiplicity.items():
            postings_iterator = self._inverted_index.get_postings_iterator(term)
            posting = next(postings_iterator, None) # None if the postingslist is empty
            cursor = self.Cursor(
                term=term,
                multiplicity=count,
                current=posting,
                postings=postings_iterator
            )
            cursors.append(cursor)

        # DAAT scoring
        k = options.hit_count if options else 10
        sieve = Sieve(k)
        alive_cursors = self._alive(cursors)

        while alive_cursors:
            frontier_doc_id, frontier_cursors_indices = self._frontier(cursors, alive_cursors)

            if len(frontier_cursors_indices) >= n:  # apply threshold
                 ranker.reset(frontier_doc_id)
                 for i in frontier_cursors_indices:
                     cursor = cursors[i]
                     ranker.update(term=cursor.term,  # adding query-term-freq x doc-term-freq
                                   multiplicity=cursor.multiplicity,
                                   posting=cursor.current)

                 score_per_doc = ranker.evaluate()   # current document's total relevancy score
                 score_cosine = score_per_doc 
                 if score_per_doc > 0:
                     sieve.sift(score_per_doc, frontier_doc_id)     # keeping top-k results
            self._advance(cursors, frontier_cursors_indices)
            alive_cursors = self._alive(cursors)

        for score, doc_id in sieve.winners():
            document = self._corpus.get_document(doc_id)
            yield self.Result(score, document)


