import json
from dataclasses import dataclass
from typing import NamedTuple, List

import numpy as np


# Константы принято писать в UPPER_CASE
@dataclass(frozen=True)
class EarthConstants:
    RADIUS: float = 6378135.0  # [m]
    GM: float = 3.986004415e+14  # [m3/s2]
    J2: float = 1.082626e-3  # Вторая зональная гармоника


CONST = EarthConstants()


class Walker(NamedTuple):
    inclination: float  # наклонение орбиты
    sats_per_plane: int  # число КА в каждой орбитальной плоскости
    plane_count: int  # число орбитальных плоскостей
    f: int  # фазовый сдвиг
    altitude: float  # высота орбиты
    max_raan: float  # максимум RAAN
    start_raan: float  # RAAN первой плоскости


class WalkerGroup(Walker):
    def get_total_sat_count(self) -> int:
        return self.sats_per_plane * self.plane_count

    def get_initial_elements(self) -> np.ndarray:
        start_raan_rad = np.deg2rad(self.start_raan)
        max_raan_rad = np.deg2rad(self.max_raan)
        inclination_rad = np.deg2rad(self.inclination)
        altitude_m = self.altitude * 1000
        total_sats = self.get_total_sat_count()

        # Генерация RAAN для плоскостей
        raans = np.linspace(start_raan_rad, start_raan_rad + max_raan_rad, self.plane_count + 1)
        raans = raans[:-1] % (2 * np.pi)

        elements = np.zeros((total_sats, 6))
        idx = 0

        for raan_idx, raan in enumerate(raans):
            for sat_idx in range(self.sats_per_plane):
                sma = CONST.RADIUS + altitude_m
                # Аргумент широты (Argument of Latitude)
                aol = 2 * np.pi * (sat_idx / self.sats_per_plane + self.f * raan_idx / total_sats)

                # [SMA, ecc, arg_perigee, RAAN, Inc, AOL]
                elements[idx, :] = [sma, 0, 0, raan, inclination_rad, aol]
                idx += 1

        return elements


class Constellation:
    def __init__(self, name_code: str, config_path: str = './../ConstellationsTest.json'):
        self.total_sat_count = 0
        self.groups: List[WalkerGroup] = []
        self.elements = []
        self.state_eci = []
        self.config_path = config_path
        self.load_from_config(name_code)

    def load_from_config(self, name_code: str):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл конфигурации {self.config_path} не найден.")

        found = False
        for entry in json_data:
            if entry['name'].lower() == name_code.lower():
                print(f"Загружена группировка {name_code}")

                for walker_params in entry['Walkers']:
                    # Распаковка параметров
                    group = WalkerGroup(*walker_params)
                    self.groups.append(group)
                    self.total_sat_count += group.get_total_sat_count()

                found = True
                break

        if not found:
            raise ValueError(f'Группировка "{name_code}" не найдена в файле.')

    def get_initial_state(self):
        self.elements = np.zeros((self.total_sat_count, 6))
        shift = 0

        for group in self.groups:
            count = group.get_total_sat_count()
            ending = shift + count
            self.elements[shift:ending, :] = group.get_initial_elements()
            shift = ending

    def propagate_j2(self, epochs: list):
        self.state_eci = np.zeros((self.total_sat_count, 3, len(epochs)))

        # Векторизованные операции NumPy (это хорошо, здесь менять нечего, кроме имен)
        inclination = self.elements[:, 4]
        sma = self.elements[:, 0]
        raan0 = self.elements[:, 3]
        aol0 = self.elements[:, 5]

        # Разбиваем длинное выражение для читаемости
        j2_term = CONST.J2 * (CONST.RADIUS / sma) ** 2
        mean_motion = np.sqrt(CONST.GM / sma ** 3)

        raan_precession_rate = -1.5 * (mean_motion * j2_term) * np.cos(inclination)

        draconic_omega = mean_motion * (1 - 1.5 * j2_term) * (1 - 4 * np.cos(inclination) ** 2)

        for i, epoch in enumerate(epochs):
            aol = aol0 + epoch * draconic_omega
            raan_omega = raan0 + epoch * raan_precession_rate

            epoch_state_x = sma * (
                        np.cos(aol) * np.cos(raan_omega) - np.sin(aol) * np.cos(inclination) * np.sin(raan_omega))
            epoch_state_y = sma * (
                        np.cos(aol) * np.sin(raan_omega) + np.sin(aol) * np.cos(inclination) * np.cos(raan_omega))
            epoch_state_z = sma * (np.sin(aol) * np.sin(inclination))

            self.state_eci[:, :, i] = np.array([epoch_state_x, epoch_state_y, epoch_state_z]).T
