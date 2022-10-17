
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List


class KmeansMatchingFunction:

    def __init__(self,
                 class_factor: int = 10,
                 color_factor: int = 5,
                 position_factor: int = 2,
                 size_factor: int = 5,
                 time_factor: int = 1
                 ) -> None:

        self.class_factor = class_factor
        self.color_factor = color_factor
        self.position_factor = position_factor
        self.size_factor = size_factor
        self.time_factor = time_factor

        # generate training data
        data = self.generate_data(step=0.01,
                                  class_factor=class_factor,
                                  color_factor=color_factor,
                                  position_factor=position_factor,
                                  size_factor=size_factor,
                                  time_factor=time_factor)

        # transform the data
        self.pca = PCA(2)
        df = self.pca.fit_transform(data)

        # train the model
        self.model = KMeans(n_clusters=2)
        self.model.fit(df)

        # cluster of matches
        self.matches_cluster = self.model.predict(
            self.pca.transform(
                [[
                    1 * self.class_factor,
                    1 * self.color_factor,
                    1 * self.position_factor,
                    1 * self.size_factor,
                    1 * self.time_factor
                ]]
            ))[0]

    def generate_data(self,
                      step: float = 0.1,
                      class_factor: int = 10,
                      color_factor: int = 5,
                      position_factor: int = 2,
                      size_factor: int = 5,
                      time_factor: int = 1
                      ) -> List[List[float]]:

        data = [[0, 0, 0, 0, 0]]
        index = 0

        for i in range(1, int(5/step) + 1):
            last = data[i - 1]

            new = []
            for ele in last:
                new.append(ele)

            def update(new, index, factor):
                new[index] = round(new[index] + step * factor, 2)

                if new[index] == factor:
                    index += 1

                return index

            # class
            if index == 0:
                index = update(new, index, class_factor)

            # color
            elif index == 1:
                index = update(new, index, color_factor)

            # position
            elif index == 2:
                index = update(new, index, position_factor)

            # size
            elif index == 3:
                index = update(new, index, size_factor)

            # time
            elif index == 4:
                index = update(new, index, time_factor)

            data.append(new)

        return data

    def match(self,
              class_similarity: float,
              color_histogram_similarity: float,
              position_similarity: float,
              size_similarity: float,
              last_time_seen_similarity: float
              ) -> float:

        prediction = self.model.predict(self.pca.transform(
            [[
                class_similarity * self.class_factor,
                color_histogram_similarity * self.color_factor,
                position_similarity * self.position_factor,
                size_similarity * self.size_factor,
                last_time_seen_similarity * self.time_factor
            ]]
        ))[0]

        return float(prediction == self.matches_cluster)
