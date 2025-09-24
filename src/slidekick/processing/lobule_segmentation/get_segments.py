import pickle
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from matplotlib import pyplot as plt

# copy of https://github.com/matthiaskoenig/zonation-image-analysis/blob/develop/src/zia/pipeline/pipeline_components/algorithm/segementation/get_segments.py
# SPEEDUPS
# - self.pixels is a set[(row, col)] for O(1) membership/removal
# - get_loop_end uses set membership instead of nested any()
# - is_ortho_and_aligned uses scalar math (no small NumPy arrays)
# CHANGES
# - Dead-end policy: segments that reach 0 next pixels are DROPPED (not finished) to “cut back to last split”.
# - walk_segment fallthroughs (after merge checks) now DROP partials when nothing remains to extend.
# - Diagonal-path fallthrough does the same DROP to avoid keeping open tails.
# - check_connected_segments signature aligned with call sites:
#   (connected_segments, orig_pixel, segment, connected_pixels)
# - check_connected_segments only marks neighbor segments finished up to their tail
#   and QUEUES new growth from their tails; it does not force-finish the current segment.
# - Small-gap closure near open ends: before dropping, probe a radius-2 ring for a single pixel
#   or queued-tail and bridge it (mirrors segments_to_mask’s “close tiny gaps” effect).

class LineSegmentsFinder:
    def __init__(self, pixels: List[Tuple[int, int]], image_shape: Tuple[int, int]):
        # store pixels as a SET of tuples -> O(1) membership/removal
        self.pixels = set(pixels)
        self.image_shape = image_shape
        self.segments_finished = []
        self.segments_to_do = []
        self.nodes = []

    def get_neighbors(self, pixel: Tuple[int, int], ortho=True):
        """
        gets the neighboring pixels for a pixel
        @param pixel: central pixel
        @param ortho: if True orthogonal neighbors are returned, else diagonal
        @return: neighboring pixels
        """
        neighbors = []
        h, w = pixel
        ih, iw = self.image_shape
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0)] if ortho else [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dh, dw in directions:
            nh, nw = h + dh, w + dw
            if 0 <= nh < ih and 0 <= nw < iw:
                neighbors.append((nh, nw))
        return neighbors

    def get_connected_pixels(self, neighbors: List[Tuple[int, int]], remove=True) -> List[Tuple[int, int]]:
        """
        finds the connected pixels by checking if the given neighboring pixel is in the pixels list.
        @param neighbors: the neighboring pixels
        @param remove: if True, the found pixels are removed from the pixels list
        @return: list of found connected pixels.
        """
        # set membership -> O(1) each
        n_pixels = [n for n in neighbors if n in self.pixels]
        if remove:
            for n_pixel in n_pixels:
                # set.remove is O(1)
                self.pixels.remove(n_pixel)
        return n_pixels

    def get_loop_end(self, neighbors: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        finds the connected segments by checking if the neighboring pixel is equal to the last pixel
        of the segments in the segments to do list
        @param neighbors: neighboring pixels, which might be the end of another segment
        @return: list of the found connected segments
        """
        # speed: use set membership rather than any([...])
        nset = set(neighbors)
        return [s for s in self.segments_to_do if s and (s[-1] in nset)]

    def _neighbors_radius(self, pixel: Tuple[int, int], r: int) -> List[Tuple[int, int]]:
        """Square (Chebyshev) ring neighbors at exact radius r within image bounds."""
        h, w = pixel
        ih, iw = self.image_shape
        out = []
        for dh in range(-r, r+1):
            for dw in range(-r, r+1):
                if max(abs(dh), abs(dw)) != r:
                    continue
                nh, nw = h + dh, w + dw
                if 0 <= nh < ih and 0 <= nw < iw:
                    out.append((nh, nw))
        return out

    def _try_gap_bridge(self, segment: List[Tuple[int, int]], max_r: int = 5) -> bool:
        """
        Small-gap closure near an open end. Mirrors segments_to_mask's 3x3 dilation idea.
        If no 8-neighbor continuation exists, probe a ring at radius 2 and, if exactly one
        candidate pixel or queued-tail is found, extend to it and continue recursion.
        Returns True if the segment was extended and recursion continued.
        """
        this_pixel = segment[-1]
        prev = segment[-2] if len(segment) >= 2 else None
        # scan radius 2 only; radius 1 is already handled by ortho/diag neighbors
        cands = self._neighbors_radius(this_pixel, max_r)
        # heuristic: prefer aligned with prev if present
        def aligned(p):
            if prev is None:
                return True
            dh1 = this_pixel[0] - prev[0]
            dw1 = this_pixel[1] - prev[1]
            dh2 = p[0] - this_pixel[0]
            dw2 = p[1] - this_pixel[1]
            return (dh1 == 0 and dh2 == 0) or (dw1 == 0 and dw2 == 0) or (abs(dh1) == abs(dw1) and abs(dh2) == abs(dw2))
        pix_hits = [p for p in cands if p in self.pixels]
        seg_hits = [s for s in self.segments_to_do if s and (s[-1] in cands)]
        # unique target logic
        target_pixel = None
        if len(pix_hits) + len(seg_hits) == 1:
            if len(pix_hits) == 1:
                target_pixel = pix_hits[0]
                self.pixels.remove(target_pixel)
                segment.append(target_pixel)
                self.walk_segment(segment)
                return True
            else:
                # attach to the queued segment tail
                neighbor_seg = seg_hits[0]
                self.extend_segment(this_pixel, segment, [], [neighbor_seg])
                return True
        # if multiple, try pick single aligned pixel
        aligned_hits = [p for p in pix_hits if aligned(p)]
        if len(aligned_hits) == 1 and len(seg_hits) == 0:
            self.pixels.remove(aligned_hits[0])
            segment.append(aligned_hits[0])
            self.walk_segment(segment)
            return True
        return False

    def walk_segment(self, segment: List[Tuple[int, int]]) -> None:
        """
        Single-step walker. Modified fallthroughs:
        - If nothing remains to extend after merge checks, DROP the partial
          instead of finishing it, so we cut back to the last split.
        """
        this_pixel = segment[-1]

        n_ortho = self.get_neighbors(this_pixel)
        n_diag = self.get_neighbors(this_pixel, ortho=False)

        # exclude where we came from
        if len(segment) >= 2:
            prev = segment[-2]
            if prev in n_ortho:
                n_ortho.remove(prev)
            if prev in n_diag:
                n_diag.remove(prev)

        orthogonally_connected = self.get_connected_pixels(n_ortho)
        ortho_connected_segments = self.get_simple_connected_segments(n_ortho, orthogonally_connected)
        ortho_connected_segments = self.filter_connected_segments(ortho_connected_segments, segment)

        # If exactly one continuation overall, probe potential diagonal branch
        if (len(orthogonally_connected) + len(ortho_connected_segments)) == 1:
            if len(orthogonally_connected) == 1:
                connected_pixel = orthogonally_connected[0]
            else:
                connected_pixel = ortho_connected_segments[0][-1]

            potential_branch = self.get_potential_branching_pixel(
                segment[-2] if len(segment) >= 2 else this_pixel,
                connected_pixel,
                n_diag
            )
            if potential_branch is not None:
                # if pixel exists, pull it now; otherwise attach the queued segment that ends there
                if potential_branch in self.pixels:
                    orthogonally_connected.append(potential_branch)
                    self.pixels.remove(potential_branch)
                else:
                    diag_seg = list(filter(lambda x: x[-1] == potential_branch, self.segments_to_do))
                    ortho_connected_segments.extend(diag_seg)

        # Handle orthogonal continuations or merges
        if len(orthogonally_connected) > 0 or len(ortho_connected_segments) > 0:
            finished_segments = self.check_connected_segments(
                ortho_connected_segments, this_pixel, segment, orthogonally_connected
            )

            # If nothing remains after merge checks -> DROP the partial (cut back to split)
            if len(orthogonally_connected) == 0 and len(ortho_connected_segments) == 0:
                # small-gap closure attempt before dropping
                if self._try_gap_bridge(segment):
                    return
                return

            # If merges produced finished pieces, store them
            if finished_segments:
                self.segments_finished.extend(finished_segments)

            # If there is exactly one pixel continuation and no attached segment, extend and recurse
            if len(orthogonally_connected) == 1 and len(ortho_connected_segments) == 0:
                segment.append(orthogonally_connected[0])
                self.walk_segment(segment)
                return

            # Otherwise delegate to unified extender (handles merges and forks)
            self.extend_segment(this_pixel, segment, orthogonally_connected, ortho_connected_segments)
            return

        # No orthogonal continuation. Try diagonal branches
        branches = self.get_branching_neighbors(n_diag, orthogonally_connected)
        if len(branches) > 0:
            for branch in branches:
                if branch in self.pixels:
                    self.pixels.remove(branch)
                    self.segments_to_do.append([this_pixel, branch])
            self.nodes.append(this_pixel)
            return

        diagonally_connected = self.get_connected_pixels(n_diag)
        diagonal_connected_segments = self.get_simple_connected_segments(n_diag, diagonally_connected)
        diagonal_connected_segments = self.filter_connected_segments(diagonal_connected_segments, segment)

        if len(diagonally_connected) > 0 or len(diagonal_connected_segments) > 0:
            finished_segments = self.check_connected_segments(
                diagonal_connected_segments, this_pixel, segment, diagonally_connected
            )

            # If nothing left after checks → DROP partial
            if len(diagonally_connected) == 0 and len(diagonal_connected_segments) == 0:
                if self._try_gap_bridge(segment):
                    return
                return

            if finished_segments:
                self.segments_finished.extend(finished_segments)

            if len(diagonally_connected) == 1 and len(diagonal_connected_segments) == 0:
                segment.append(diagonally_connected[0])
                self.walk_segment(segment)
                return

            self.extend_segment(this_pixel, segment, diagonally_connected, diagonal_connected_segments)
            return

        # Truly nothing to do in any direction -> try small-gap closure, else DROP partial
        if self._try_gap_bridge(segment):
            return
        return

    def check_connected_segments(self,
                                 connected_segments: List[List[Tuple[int, int]]],
                                 orig_pixel: Tuple[int, int],
                                 segment: List[Tuple[int, int]],
                                 connected_pixels: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        Re-check queued neighbor segments at a junction and re-queue their growth so we don't lose branches.

        Args:
            connected_segments: segments whose tail touches the current pixel neighborhood.
            orig_pixel:         pixel at which we are evaluating the junction (the 'this_pixel' in caller).
            segment:            current growing segment (not modified here).
            connected_pixels:   current list of pixel continuations (not used except for symmetry with caller).

        Returns:
            finished_segments: segments that should be considered complete up to their current tail.
                               New growth from their tails is queued into self.segments_to_do.
        """
        finished_segments: List[List[Tuple[int, int]]] = []

        for seg2 in connected_segments:
            # Tail of neighbor segment
            if len(seg2) == 0:
                continue
            tail = seg2[-1]
            prev = seg2[-2] if len(seg2) >= 2 else orig_pixel

            # Neighbors around tail, exclude where it came from and the current origin pixel
            n_ortho = self.get_neighbors(tail)
            n_diag = self.get_neighbors(tail, ortho=False)

            if prev in n_ortho:
                n_ortho.remove(prev)
            if prev in n_diag:
                n_diag.remove(prev)
            if orig_pixel in n_ortho:
                n_ortho.remove(orig_pixel)
            if orig_pixel in n_diag:
                n_diag.remove(orig_pixel)

            # Available continuations from seg2's tail
            cont_pixels_ortho = self.get_connected_pixels(n_ortho)
            cont_segs_ortho = self.get_simple_connected_segments(n_ortho, cont_pixels_ortho)
            cont_segs_ortho = self.filter_connected_segments(cont_segs_ortho, seg2)

            # Optional diagonal probe for branching patterns identical to the caller logic
            if (len(cont_pixels_ortho) + len(cont_segs_ortho)) == 1:
                if len(cont_pixels_ortho) == 1:
                    probe_pixel = cont_pixels_ortho[0]
                else:
                    probe_pixel = cont_segs_ortho[0][-1]

                pot = self.get_potential_branching_pixel(prev, probe_pixel, n_diag)
                if pot is not None:
                    if pot in self.pixels:
                        cont_pixels_ortho.append(pot)
                        self.pixels.remove(pot)
                    else:
                        diag_seg = list(filter(lambda x: x[-1] == pot, self.segments_to_do))
                        cont_segs_ortho.extend(diag_seg)

            # Also consider diagonal-only continuations when no ortho options
            cont_pixels_diag = []
            cont_segs_diag = []
            if len(cont_pixels_ortho) == 0 and len(cont_segs_ortho) == 0:
                branches = self.get_branching_neighbors(n_diag, cont_pixels_ortho)
                for b in branches:
                    if b in self.pixels:
                        self.pixels.remove(b)
                        cont_pixels_diag.append(b)
                extra = self.get_connected_pixels(n_diag)
                cont_pixels_diag.extend(extra)
                cont_segs_diag = self.get_simple_connected_segments(n_diag, extra)
                cont_segs_diag = self.filter_connected_segments(cont_segs_diag, seg2)

            has_continuation = (
                len(cont_pixels_ortho) + len(cont_segs_ortho) + len(cont_pixels_diag) + len(cont_segs_diag)
            ) > 0

            # If seg2 can continue from its tail, mark it finished up to tail and queue new growth
            if has_continuation:
                finished_segments.append(seg2)

                # Queue orthogonal pixel continuations
                for p in cont_pixels_ortho:
                    self.segments_to_do.append([tail, p])

                # Queue orthogonal segment continuations
                for s in cont_segs_ortho:
                    self.segments_to_do.append([tail, s[-1]])

                # Queue diagonal pixel continuations
                for p in cont_pixels_diag:
                    self.segments_to_do.append([tail, p])

                # Queue diagonal segment continuations
                for s in cont_segs_diag:
                    self.segments_to_do.append([tail, s[-1]])

            # If no continuation, we do nothing here; caller will decide whether to drop or finish its own segment.

        return finished_segments

    def get_diagonally_branching_pixel(self, ortho: Tuple[int, int], diag: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return list(filter(lambda d: self.is_branch(ortho, d), diag))

    def get_potential_branching_pixel(self, origin: Tuple[int, int], ortho: Tuple[int, int], diag: List[Tuple[int, int]]) -> Optional[
        Tuple[int, int]]:
        potential_branches = list(filter(lambda d: self.is_branch1(d, origin, ortho), diag))
        if len(potential_branches) != 1:
            return None
        return potential_branches[0]

    def is_branch1(self, test: Tuple[int, int], origin: Tuple[int, int], ortho: Tuple[int, int]) -> bool:
        h, w = test  # test pixel
        h1, w1 = origin  # origin
        h2, w2 = ortho  # ortho connection

        dh1 = abs(h1 - h)
        dw1 = abs(w1 - w)

        dh2 = abs(h2 - h)
        dw2 = abs(w2 - w)

        d1 = dh1 + dw1
        d2 = dh2 + dw2

        if d2 == 3 and (d1 in [2, 3]):
            return True
        return False

    def is_branch(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        h1, w1 = p1
        h2, w2 = p2

        dh = abs(h2 - h1)
        dw = abs(w2 - w1)

        d = dh + dw

        if d == 1:
            return False
        elif d == 3:
            return True
        else:
            raise ValueError(f"Value {d} is not allowed.")

    def get_simple_connected_segments(self,
                                      neighbors: List[Tuple[int, int]],
                                      connected_pixels: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        remaining_n = list(filter(lambda x: x not in connected_pixels, neighbors))
        return self.get_loop_end(remaining_n)

    def extend_segment(self,
                       this_pixel: Tuple[int, int],
                       segment: List[Tuple[int, int]],
                       connected_pixels: List[Tuple[int, int]],
                       connected_segments: List[List[Tuple[int, int]]]) -> None:
        """
        Merge-first policy. If only one attached segment and no pixel, merge it.
        Otherwise fork. When nothing to merge, defer to pixel-only handler which
        enforces the "drop dead tail" rule.
        """
        # No connected segments -> handle purely by pixel logic
        if len(connected_segments) == 0:
            self.extend_segment_pixels_only(this_pixel, segment, connected_pixels)
            return

        # Exactly one connected segment and no pixel -> merge and finish
        if len(connected_segments) == 1 and len(connected_pixels) == 0:
            seg2 = connected_segments[0]
            # ensure seg2 is oriented away from this_pixel
            if len(seg2) >= 2 and seg2[-2] == this_pixel:
                # already oriented
                merged = segment + seg2[-1:]
            else:
                # reverse so tail touches this_pixel
                seg2 = list(reversed(seg2))
                merged = segment + seg2[-1:]
            self.segments_finished.append(merged)
            return

        # Otherwise it's a junction: end current here, end any attached segments at the node,
        # and queue all new branches starting from the node.
        self.nodes.append(this_pixel)
        self.segments_finished.append(segment)
        for seg2 in connected_segments:
            # normalize to end at node
            if seg2[-1] != this_pixel:
                if seg2[0] == this_pixel:
                    seg2 = list(reversed(seg2))
                # if neither end is the node, leave as-is; upstream code should only pass tails here
            self.segments_finished.append(seg2)
        for p in connected_pixels:
            self.segments_to_do.append([this_pixel, p])
        return

    def extend_segment_pixels_only(self,
                                   this_pixel: Tuple[int, int],
                                   segment: List[Tuple[int, int]],
                                   connected_pixels: List[Tuple[int, int]]) -> None:
        """
        Cut back to the last split:
        - 0 next pixels  -> drop this partial (do NOT finish).
        - 1 next pixel   -> extend and continue walking.
        - >1 next pixels -> mark node, finish current partial here, and fork.
        """
        # 0 next pixels → dead tail; drop it so we "return to last split"
        if len(connected_pixels) == 0:
            return

        # 1 next pixel → extend inline and continue
        if len(connected_pixels) == 1:
            segment.append(connected_pixels[0])
            self.walk_segment(segment)
            return

        # >1 next pixels → split: finish current partial and queue branches
        self.nodes.append(this_pixel)
        self.segments_finished.append(segment)
        for p in connected_pixels:
            self.segments_to_do.append([this_pixel, p])
        return

    def process_segments(self):
        """
        Processes the segments in the segments to do list until as this list is not empty.
        """
        while len(self.segments_to_do) != 0:
            next_segment = self.segments_to_do.pop()
            self.walk_segment(next_segment)

    def initialize(self, this_pixel: Tuple[int, int]) -> None:
        """
        initializes the segmentation algorithm. It takes the given pixel and finds connected pixel to
        initialize the segments to do list
        @param this_pixel: the first pixel.
        """
        n_ortho = self.get_neighbors(this_pixel)
        n_diagonal = self.get_neighbors(this_pixel, ortho=False)

        # get connected pixels
        orthogonally_connected = self.get_connected_pixels(n_ortho)

        if len(orthogonally_connected) > 0:

            if len(orthogonally_connected) == 1:
                diagonally_connected = self.get_connected_pixels(n_diagonal, False)
                branches = self.get_diagonally_branching_pixel(orthogonally_connected[0], diagonally_connected)
                if len(branches) != 0:
                    for branch in branches:
                        if branch in self.pixels:
                            self.pixels.remove(branch)
                            orthogonally_connected.append(branch)

            for p in orthogonally_connected:
                self.segments_to_do.append([this_pixel, p])

            self.nodes.append(this_pixel)

            return

        diagonally_connected = self.get_connected_pixels(n_diagonal)

        if len(diagonally_connected) > 0:
            for p in diagonally_connected:
                self.segments_to_do.append([this_pixel, p])
            self.nodes.append(this_pixel)

        return

    def run(self) -> List[List[Tuple[int, int]]]:
        while len(self.pixels) > 0:
            pixel = self.pixels.pop()
            self.initialize(pixel)
            self.process_segments()

        # prune entire dead branches
        self.prune_dead_branches()

        return self.segments_finished

    def filter_connected_segments(self, connected_segments: List[List[Tuple[int, int]]], segment: List[Tuple[int, int]]) -> List[
        List[Tuple[int, int]]]:
        """
        Filters connected segments to avoid small circles:
        Case 1: Two segments of length 2 that have the same origin.
        Case 2: Two parallely aligned segments of length 2. This happens if there are 4 pixels in a square layout.
        @param connected_segments: the connected segments to check
        @param segment: the current segment in progress
        @return: the filtered list of valid segments
        """
        return [s for s in connected_segments if len(segment) > 2 or (s[0] != segment[0] and not self.is_ortho_and_aligned(s, segment))]

    def is_ortho_and_aligned(self, s1, s2) -> bool:
        # scalar arithmetic (faster than tiny NumPy arrays)
        dx1 = s1[1][0] - s1[0][0]
        dy1 = s1[1][1] - s1[0][1]
        dx2 = s2[1][0] - s2[0][0]
        dy2 = s2[1][1] - s2[0][1]
        cross = dx1 * dy2 - dy1 * dx2
        d = abs(s1[0][0] - s2[0][0]) + abs(s1[0][1] - s2[0][1])
        return (cross == 0) and (d == 1)

    def get_branching_neighbors(self, neighbors_diag: List[Tuple[int, int]], connected_pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Finds diagonally branching neighbors
        @param neighbors_diag: diagonally neighboring pixels
        @param connected_pixels: orthogonally connected pixels
        @return: list of neighbors. Max length of list is one.
        """
        if len(connected_pixels) == 0 or len(connected_pixels) > 1:
            return []
        potential_branches = [n for n in neighbors_diag if self.is_branch(connected_pixels[0], n)]
        return potential_branches

    def prune_dead_branches(self) -> None:
        """
        Recursively remove branches that end in nothing.
        Uses degrees computed from segment endpoints only.
        Keeps loops. O(#segments + #endpoints).
        """
        if not self.segments_finished:
            return

        from collections import defaultdict, deque

        # endpoints list
        ends = [(seg[0], seg[-1]) for seg in self.segments_finished]

        # degree from segments themselves
        deg = defaultdict(int)
        touch = defaultdict(set)  # point -> set(segment idx)
        for i, (a, b) in enumerate(ends):
            deg[a] += 1
            deg[b] += 1
            touch[a].add(i)
            touch[b].add(i)

        removed = [False] * len(self.segments_finished)

        # leaf = point with degree 1
        q = deque([p for p, d in deg.items() if d == 1])

        while q:
            p = q.popleft()
            if deg.get(p, 0) != 1:
                continue
            # remove the single incident segment
            s_idx = next(iter(touch[p]))
            if removed[s_idx]:
                continue
            removed[s_idx] = True

            a, b = ends[s_idx]
            other = b if p == a else a

            # update degree and adjacency
            touch[p].remove(s_idx)
            deg[p] -= 1
            if deg[p] == 0:
                del deg[p]
                del touch[p]

            if s_idx in touch[other]:
                touch[other].remove(s_idx)
                deg[other] -= 1
                if deg[other] == 1:
                    q.append(other)
                if deg[other] == 0:
                    del deg[other]
                    del touch[other]

        # keep non-removed segments
        self.segments_finished = [seg for i, seg in enumerate(self.segments_finished) if not removed[i]]


def segment_thinned_image(image: np.ndarray, write=False, report_path: Path = None) -> List[List[Tuple[int, int]]]:
    pixels = np.argwhere(image == 255)
    pixels = [tuple(coords) for coords in pixels]

    segmenter = LineSegmentsFinder(pixels, image.shape[:2])
    segements_finished = segmenter.run()

    if write:
        with open("segmenter.pickle", "wb") as f:
            pickle.dump(segmenter, f)

    if report_path is not None:
        report_path.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(dpi=300)
        ax.invert_yaxis()
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)  # show skeleton as background
        colors = np.random.rand(len(segements_finished), 3)  # Random RGB rows in [0,1]
        for i, line in enumerate(segements_finished):
            x, y = zip(*line)  # (row, col) -> plot as (y, x)
            ax.plot(y, x, linewidth=1, color=colors[i], alpha=0.9)
        ax.axis("off")
        fig.savefig(report_path / "line_segmentation.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    return segements_finished
