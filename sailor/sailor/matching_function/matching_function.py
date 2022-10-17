
class MatchingFunction:

    def __init__(self,
                 color_factor: int = 5,
                 position_factor: int = 2,
                 size_factor: int = 5,
                 time_factor: int = 1
                 ) -> None:

        self.color_factor = color_factor
        self.position_factor = position_factor
        self.size_factor = size_factor
        self.time_factor = time_factor

    def match(self,
              class_similarity: float,
              color_histogram_similarity: float,
              position_similarity: float,
              size_similarity: float,
              last_time_seen_similarity: float
              ) -> float:

        if class_similarity == 0:
            return 0.0

        return (
            color_histogram_similarity * self.color_factor +
            position_similarity * self.position_factor +
            size_similarity * self.size_factor +
            last_time_seen_similarity * self.time_factor
        ) / (
            self.color_factor +
            self.position_factor +
            self.size_factor +
            self.time_factor
        )
