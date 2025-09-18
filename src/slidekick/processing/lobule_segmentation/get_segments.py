import pickle
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from matplotlib import pyplot as plt

# simple copy of https://github.com/matthiaskoenig/zonation-image-analysis/blob/develop/src/zia/pipeline/pipeline_components/algorithm/segementation/get_segments.py
# SPEEDUPS
# - self.pixels is a set[(row, col)] for O(1) membership/removal
# - get_loop_end uses set membership instead of nested any()
# - is_ortho_and_aligned uses scalar math (no small NumPy arrays)


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

    def walk_segment(self, segment: List[tuple[int, int]]) -> None:
        """
        should take a segment until there is no or multiple connected pixels
        """
        this_pixel = segment[-1]
        n_ortho = self.get_neighbors(this_pixel)
        n_diagonal = self.get_neighbors(this_pixel, ortho=False)

        # remove already visited from neighbors
        if segment[-2] in n_ortho:
            n_ortho.remove(segment[-2])
        if segment[-2] in n_diagonal:
            n_diagonal.remove(segment[-2])

        # get orthogonally connected pixels
        orthogonally_connected = self.get_connected_pixels(n_ortho)
        # orthogonally connected segments
        ortho_connected_segments = self.get_simple_connected_segments(n_ortho, orthogonally_connected)
        ortho_connected_segments = self.filter_connected_segments(ortho_connected_segments, segment)

        # if only one orthogonally connected pixel or segment is found there might be a diagonally branching pixel or segment
        if (len(orthogonally_connected) + len(ortho_connected_segments)) == 1:
            if len(orthogonally_connected) == 1:
                connected_pixel = orthogonally_connected[0]
            else:
                connected_pixel = ortho_connected_segments[0][-1]

            # find the diagonal neighbor that is far from the orthogonally connected pixel
            potential_branch = self.get_potential_branching_pixel(segment[-2], connected_pixel, n_diagonal)

            # if that neighbor exists, check if it is another pixel or another segment end
            if potential_branch is not None:
                if potential_branch in self.pixels:
                    orthogonally_connected.append(potential_branch)
                    self.pixels.remove(potential_branch)
                else:
                    diag_segment = list(filter(lambda x: x[-1] == potential_branch, self.segments_to_do))
                    ortho_connected_segments.extend(diag_segment)

        # if one entry exists, extend the segment otherwise check for diagonal pixels
        if len(orthogonally_connected) > 0 or len(ortho_connected_segments) > 0:
            finished_segments = self.check_connected_segments(ortho_connected_segments, this_pixel, segment)
            if len(finished_segments) != 0:
                for s in finished_segments:
                    # safe remove (list) – keep structure
                    if s in self.segments_to_do:
                        self.segments_to_do.remove(s)
                    self.segments_finished.append(s)
                    if s in ortho_connected_segments:
                        ortho_connected_segments.remove(s)
                if len(orthogonally_connected) == 0 and len(ortho_connected_segments) == 0:
                    segment.append(finished_segments[0][-1])
                    self.segments_finished.append(segment)
                    return
                else:
                    orthogonally_connected.append(finished_segments[0][-1])
            self.extend_segment(this_pixel, segment, orthogonally_connected, ortho_connected_segments)
            return

        diagonally_connected = self.get_connected_pixels(n_diagonal)
        diag_connected_segments = self.get_simple_connected_segments(n_diagonal, diagonally_connected)
        diag_connected_segments = self.filter_connected_segments(diag_connected_segments, segment)
        finished_segments = self.check_connected_segments(diag_connected_segments, this_pixel, segment)
        if len(finished_segments) != 0:
            for s in finished_segments:
                if s in self.segments_to_do:
                    self.segments_to_do.remove(s)
                self.segments_finished.append(s)
                if s in diag_connected_segments:
                    diag_connected_segments.remove(s)
            if len(diagonally_connected) == 0 and len(diag_connected_segments) == 0:
                segment.append(finished_segments[0][-1])
                self.segments_finished.append(segment)
                return
            else:
                diagonally_connected.append(finished_segments[0][-1])
        self.extend_segment(this_pixel, segment, diagonally_connected, diag_connected_segments)
        return

    def check_connected_segments(self, connected_segments: List[List[Tuple[int, int]]], orig_pixel, segment: List[Tuple[int, int]]) -> List[
        List[Tuple[int, int]]]:
        """
        this method is needed to check if encountered segments have connected pixels or segments. These would otherwise get lost
        because the segment is removed from the segments to do list.
        @param connected_segments: list of connected segments
        @param orig_pixel: the pixel of which the neighbor is adjacent to the connected segments
        """
        finished = []
        for connected_segment in connected_segments:
            this_pixel = connected_segment[-1]
            n_ortho = self.get_neighbors(this_pixel)
            n_diagonal = self.get_neighbors(this_pixel, ortho=False)

            # remove already visited from neighbors
            if orig_pixel in n_ortho:
                n_ortho.remove(orig_pixel)
            if orig_pixel in n_diagonal:
                n_diagonal.remove(orig_pixel)

            if connected_segment[0] in n_ortho:
                n_ortho.remove(connected_segment[0])
            if connected_segment[0] in n_diagonal:
                n_diagonal.remove(connected_segment[0])

            orthogonally_connected = self.get_connected_pixels(n_ortho)
            if len(orthogonally_connected) != 0:
                finished.append(connected_segment)
                for con_pixel in orthogonally_connected:
                    self.segments_to_do.append([connected_segment[-1], con_pixel])
                continue

            ortho_connected_segments = self.get_simple_connected_segments(n_ortho, orthogonally_connected)
            if len(ortho_connected_segments) != 0:
                finished.append(connected_segment)
                for con_seg in ortho_connected_segments:
                    self.segments_to_do.append([connected_segment[-1], con_seg[-1]])
                continue

            diag_connected = self.get_connected_pixels(n_diagonal)
            if len(diag_connected) != 0:
                finished.append(connected_segment)
                for con_pixel in diag_connected:
                    self.segments_to_do.append([connected_segment[-1], con_pixel])
                continue

            connected_segments2 = self.get_simple_connected_segments(n_diagonal, diag_connected)
            if len(connected_segments2) != 0:
                finished.append(connected_segment)
                for con_seg in connected_segments2:
                    self.segments_to_do.append([connected_segment[-1], con_seg[-1]])
                continue

        return finished

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

    def get_connected_segments(self,
                               neighbors_ortho: List[Tuple[int, int]],
                               neighbors_diag: List[Tuple[int, int]],
                               connected_pixels: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        Gets all the orthogonally connected segments and diagonally connected segments which are not adjacent to
        any of the orthogonally connected segments.
        @param neighbors_ortho: orthogonally connected neighboring pixels
        @param neighbors_diag: diagonally connected neighboring pixels
        @param connected_pixels: actual connected pixels.
        @return: list of connected segments
        """
        branching_n_diag = self.get_branching_neighbors(neighbors_diag, connected_pixels)
        # get the potential branching diagonal neighbors
        remaining_n = list(filter(lambda x: x not in connected_pixels, neighbors_ortho + branching_n_diag))
        return self.get_loop_end(remaining_n)

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
        Extends or finishes the current segment based on the connected pixels and the connected segments
        @param this_pixel: the current pixel
        @param segment: the current segment
        @param connected_pixels: pixels connected to current pixel
        @param connected_segments: segments connected to current pixel
        @return: None
        """
        # if no connection found -> branch ending in the void
        if len(connected_segments) == 0:
            self.extend_segment_pixels_only(this_pixel, segment, connected_pixels)
            return

        # if 1 connected segment is found and no connected pixel -> end by merging connected segment
        elif len(connected_segments) == 1 and len(connected_pixels) == 0:
            if connected_segments[0] in self.segments_to_do:
                self.segments_to_do.remove(connected_segments[0])
            connected_segments[0].reverse()
            segment.extend(connected_segments[0])
            self.segments_finished.append(segment)

            return

        # if more than one connection -> finish segments and add connected pixels to segments to do
        else:
            self.nodes.append(this_pixel)
            self.segments_finished.append(segment)

            for connected_segment in connected_segments:
                if connected_segment in self.segments_to_do:
                    self.segments_to_do.remove(connected_segment)
                connected_segment.append(this_pixel)
                self.segments_finished.append(connected_segment)

            for connected_pixel in connected_pixels:
                self.segments_to_do.append([this_pixel, connected_pixel])

            return

    def extend_segment_pixels_only(self,
                                   this_pixel: Tuple[int, int],
                                   segment: List[Tuple[int, int]],
                                   connected_pixels: List[Tuple[int, int]]) -> None:
        """
        Extends or finished the current segment with based on the connected pixels.
        @param this_pixel: the current pixel
        @param segment: the current segment
        @param connected_pixels: the pixels connected to the current pixel
        @return: None
        """
        if len(connected_pixels) == 0:
            self.segments_finished.append(segment)
            return

        # if one connected pixel is found -> walk on
        if len(connected_pixels) == 1:
            segment.append(connected_pixels[0])
            self.walk_segment(segment)
            return

        # if multiple connections found -> pixel is node, initialize new segments
        if len(connected_pixels) > 1:
            self.nodes.append(this_pixel)
            self.segments_finished.append(segment)

            for connected_pixel in connected_pixels:
                self.segments_to_do.append([this_pixel, connected_pixel])
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
        """
        Runs the line segmentation.
        """
        # self.pixels is a set -> .pop() gives an arbitrary pixel (fast O(1))
        while len(self.pixels) > 0:
            pixel = self.pixels.pop()
            self.initialize(pixel)
            self.process_segments()

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
