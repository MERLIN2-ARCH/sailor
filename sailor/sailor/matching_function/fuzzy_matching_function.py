
import skfuzzy.control as ctrl


class FuzzyMatchingFunction:

    def __init__(self) -> None:

        universe = [0.0, 0.5, 1.0]
        names = ["low", "medium", "high"]

        # input
        class_similarity = ctrl.Antecedent(universe, "class_similarity")
        color_histogram_similarity = ctrl.Antecedent(
            universe, "color_histogram_similarity")
        position_similarity = ctrl.Antecedent(universe, "position_similarity")
        size_similarity = ctrl.Antecedent(universe, "size_similarity")
        last_time_seen_similarity = ctrl.Antecedent(
            universe, "last_time_seen_similarity")

        class_similarity.automf(names=names)
        color_histogram_similarity.automf(names=names)
        position_similarity.automf(names=names)
        size_similarity.automf(names=names)
        last_time_seen_similarity.automf(names=names)

        # output
        mathcing = ctrl.Consequent(universe, "mathcing")
        mathcing.automf(names=names)

        # rules
        rule0 = ctrl.Rule(
            antecedent=(
                class_similarity["low"] |
                color_histogram_similarity["low"] |
                size_similarity["low"] |
                (position_similarity["low"] &
                 (last_time_seen_similarity["low"] |
                 last_time_seen_similarity["medium"]) &
                 (color_histogram_similarity["medium"] |
                 size_similarity["medium"])
                 )
            ),
            consequent=mathcing["low"],
            label="rule0")

        rule1 = ctrl.Rule(
            antecedent=(

                (class_similarity["medium"] | class_similarity["high"]) & (

                    (position_similarity["low"] &
                     last_time_seen_similarity["medium"] &
                     (color_histogram_similarity["high"] |
                      size_similarity["high"])
                     ) |

                    (position_similarity["high"] &
                     (last_time_seen_similarity["low"] |
                        last_time_seen_similarity["medium"]) &
                        (color_histogram_similarity["medium"] |
                         size_similarity["medium"])
                     )
                )
            ),
            consequent=mathcing["medium"],
            label="rule1")

        rule2 = ctrl.Rule(
            antecedent=(

                (class_similarity["medium"] | class_similarity["high"]) & (

                    (position_similarity["high"] &
                     (color_histogram_similarity["high"] |
                     size_similarity["high"])
                     ) |

                    (position_similarity["high"] &
                     last_time_seen_similarity["high"] &
                     (color_histogram_similarity["medium"] |
                     size_similarity["medium"]
                      )
                     )
                )
            ),
            consequent=mathcing["high"],
            label="rule2")

        system = ctrl.ControlSystem([rule0, rule1, rule2])
        self.sim = ctrl.ControlSystemSimulation(system)

    def match(self,
              class_similarity: float,
              color_histogram_similarity: float,
              position_similarity: float,
              size_similarity: float,
              last_time_seen_similarity: float
              ) -> float:

        self.sim.input["class_similarity"] = class_similarity
        self.sim.input["color_histogram_similarity"] = color_histogram_similarity
        self.sim.input["position_similarity"] = position_similarity
        self.sim.input["size_similarity"] = size_similarity
        self.sim.input["last_time_seen_similarity"] = last_time_seen_similarity

        self.sim.compute()
        output = self.sim.output["mathcing"]

        return float(output)
