
class MatchingFunction:

    def match(self,
              class_similarity: float,
              color_histogram_similarity: float,
              position_similarity: float,
              size_similarity: float,
              last_time_seen_similarity: float
              ) -> float:

        if class_similarity == 0:
            return 0.0

        return (color_histogram_similarity *
                size_similarity *
                position_similarity *
                last_time_seen_similarity)
