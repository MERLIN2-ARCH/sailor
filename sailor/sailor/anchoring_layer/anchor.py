# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from kant_dto import PddlObjectDto
from sailor.perceptual_layer import Percept


class Anchor:

    def __init__(self) -> None:
        self._percept: Percept = None
        self._symbol: PddlObjectDto = None

    def update(self, other: "Anchor") -> None:
        self.percept = other.percept

    @property
    def percept(self) -> Percept:
        return self._percept

    @percept.setter
    def percept(self, percept: Percept) -> None:
        self._percept = percept

    @property
    def symbol(self) -> PddlObjectDto:
        return self._symbol

    @symbol.setter
    def symbol(self, symbol: PddlObjectDto) -> None:
        self._symbol = symbol
