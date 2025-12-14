#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4: ISL Coverage Analysis (Corrected)
Исправленная версия с учетом вращения Земли (ECI->ECEF) и векторизацией.
"""

import json
import argparse
import sys
import os
import math
import numpy as np
from typing import Set, List

# Попытка импорта constellation
try:
    from constellation import Constellation, EarthConstants
except ImportError:
    print("Ошибка: Файл constellation.py не найден.")
    sys.exit(1)

# Угловая скорость вращения Земли [рад/с]
EARTH_ROTATION_RATE = 7.2921158553e-5


class Task4Solver:
    def __init__(self, gateways_file: str = './../gatewaysTest.json'):
        self.constellation = None
        self.gateways_ecef = None  # numpy array (N_gw, 3)
        self._load_gateways(gateways_file)

    def _load_gateways(self, filename: str):
        if not os.path.exists(filename):
            print(f"Предупреждение: Файл {filename} не найден.")
            self.gateways_ecef = np.empty((0, 3))
            return

        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            gw_list = []
            for item in data:
                lat, lon = np.deg2rad(item['lat']), np.deg2rad(item['lon'])
                x = EarthConstants.RADIUS * np.cos(lat) * np.cos(lon)
                y = EarthConstants.RADIUS * np.cos(lat) * np.sin(lon)
                z = EarthConstants.RADIUS * np.sin(lat)
                gw_list.append([x, y, z])
            self.gateways_ecef = np.array(gw_list)
            print(f"✓ Загружено шлюзов: {len(gw_list)}")

    def _eci_to_ecef(self, positions_eci: np.ndarray, time_sec: float) -> np.ndarray:
        """
        Перевод координат из инерциальной (ECI) во вращающуюся (ECEF) систему.
        """
        theta = EARTH_ROTATION_RATE * time_sec
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Матрица поворота вокруг оси Z (против часовой стрелки, если смотреть с севера,
        # но нам нужно перевести ECI -> ECEF, значит поворачиваем систему координат,
        # или поворачиваем точку в обратную сторону.
        # ECEF повернута относительно ECI на угол theta.
        # X_ecef = X_eci * cos(theta) + Y_eci * sin(theta)
        # Y_ecef = -X_eci * sin(theta) + Y_eci * cos(theta)

        x = positions_eci[:, 0] * cos_t + positions_eci[:, 1] * sin_t
        y = -positions_eci[:, 0] * sin_t + positions_eci[:, 1] * cos_t
        z = positions_eci[:, 2]

        return np.column_stack((x, y, z))

    def get_sat_positions(self, epoch: float, use_j2: bool) -> np.ndarray:
        """Получение позиций спутников в системе ECEF на момент epoch."""
        if use_j2:
            self.constellation.propagateJ2([epoch])
            pos_eci = self.constellation.stateEci[:, :, 0]
        else:
            # Ручной кеплеров расчет (упрощенный)
            count = self.constellation.total_sat_count
            pos_eci = np.zeros((count, 3))
            elements = self.constellation.elements

            smas = elements[:, 0]
            raans = elements[:, 3]
            incs = elements[:, 4]
            aols_0 = elements[:, 5]

            mean_motions = np.sqrt(EarthConstants.GM / smas ** 3)
            aols = aols_0 + mean_motions * epoch

            # ECI расчет
            xs = smas * (np.cos(aols) * np.cos(raans) - np.sin(aols) * np.cos(incs) * np.sin(raans))
            ys = smas * (np.cos(aols) * np.sin(raans) + np.sin(aols) * np.cos(incs) * np.cos(raans))
            zs = smas * (np.sin(aols) * np.sin(incs))
            pos_eci = np.column_stack((xs, ys, zs))

        # ВАЖНО: Перевод в ECEF, так как шлюзы и Земля вращаются
        return self._eci_to_ecef(pos_eci, epoch)

    def _get_plane_ids(self) -> np.ndarray:
        """Векторизованное получение ID плоскости для всех спутников."""
        plane_ids = np.zeros(self.constellation.total_sat_count, dtype=int)
        idx = 0
        for group_id, group in enumerate(self.constellation.groups):
            count = group.get_total_sat_count()
            sats_per_plane = group.sats_per_plane

            # Генерируем ID для спутников этой группы
            # Пример: Group 0, SatsPerPlane 20. Sats 0-19 -> Plane 0. Sats 20-39 -> Plane 1.
            # Global Plane ID = GroupID * 1000 + LocalPlaneID
            local_indices = np.arange(count)
            local_plane_ids = local_indices // sats_per_plane
            global_plane_ids = group_id * 1000 + local_plane_ids

            plane_ids[idx: idx + count] = global_plane_ids
            idx += count
        return plane_ids

    def solve(self, const_name, elev_deg, alpha_deg, grid_res, epoch, use_j2):
        # 1. Init
        try:
            self.constellation = Constellation(const_name)
            self.constellation.get_initial_state()
        except Exception as e:
            print(f"Ошибка инициализации: {e}")
            return

        # 2. Get Positions (ECEF)
        print(f"Расчет орбит (t={epoch}, J2={use_j2})...")
        sat_pos_ecef = self.get_sat_positions(epoch, use_j2)
        sat_radii = np.linalg.norm(sat_pos_ecef, axis=1)

        # 3. Find Active Planes (Vectorized)
        print(f"Определение активных плоскостей (ε ≥ {elev_deg}°)...")
        if len(self.gateways_ecef) == 0:
            print("Нет шлюзов! Покрытие = 0.")
            return

        # Матрица расстояний: (N_sats, N_gw) - может быть большой, но для Task 4 терпимо
        # Оптимизация: считаем по спутникам
        # Условие видимости: elevation >= min_elev
        # sin(el) = ((S - G) dot G) / (|S-G| * |G|) >= sin(min_elev)
        # G / |G| = G_unit (normal)

        min_sin_el = np.sin(np.deg2rad(elev_deg))
        plane_ids = self._get_plane_ids()
        active_planes = set()

        # Для ускорения не строим полную матрицу, идем по шлюзам (их обычно мало)
        gw_norms = np.linalg.norm(self.gateways_ecef, axis=1)
        gw_units = self.gateways_ecef / gw_norms[:, np.newaxis]

        # Массив флагов активности спутников
        sat_is_active = np.zeros(self.constellation.total_sat_count, dtype=bool)

        for i, gw in enumerate(self.gateways_ecef):
            gw_u = gw_units[i]
            # Векторы от шлюза ко всем спутникам
            vecs = sat_pos_ecef - gw
            dists = np.linalg.norm(vecs, axis=1)

            # Скалярное произведение вектора на нормаль (высоту)
            # Избегаем деления на ноль
            dists[dists == 0] = 0.001

            sin_els = np.sum(vecs * gw_u, axis=1) / dists

            # Условие видимости
            visible_mask = sin_els >= min_sin_el

            # Отмечаем плоскости
            visible_planes = np.unique(plane_ids[visible_mask])
            active_planes.update(visible_planes)

        active_planes = np.array(list(active_planes))
        print(f"✓ Активных плоскостей: {len(active_planes)}")

        # 4. Coverage Analysis (Vectorized)
        print(f"Расчет покрытия (Grid {grid_res}x{grid_res * 2}, α={alpha_deg}°)...")

        # Фильтруем только спутники из активных плоскостей
        active_mask = np.isin(plane_ids, active_planes)
        active_sats_pos = sat_pos_ecef[active_mask]
        active_sats_r = sat_radii[active_mask]

        if len(active_sats_pos) == 0:
            print("Нет активных спутников.")
            return

        # Генерация сетки точек (Unit vectors)
        lat_step = 180.0 / grid_res
        lon_step = 360.0 / (grid_res * 2)

        lats = np.linspace(90 - lat_step / 2, -90 + lat_step / 2, grid_res)
        lons = np.linspace(-180 + lon_step / 2, 180 - lon_step / 2, grid_res * 2)

        # Meshgrid
        lon_grid, lat_grid = np.meshgrid(np.radians(lons), np.radians(lats))

        # Перевод сетки в декартовы координаты (Unit sphere)
        # Форма (N_points, 3)
        g_x = np.cos(lat_grid) * np.cos(lon_grid)
        g_y = np.cos(lat_grid) * np.sin(lon_grid)
        g_z = np.sin(lat_grid)

        # Выпрямляем в массив точек
        grid_points = np.column_stack((g_x.ravel(), g_y.ravel(), g_z.ravel()))
        n_points = grid_points.shape[0]

        # Расчет предельных косинусов для каждого активного спутника
        # cos(theta_max)
        alpha_rad = np.deg2rad(alpha_deg)
        sin_alpha = np.sin(alpha_rad)

        # Геометрический предел видимости (horizon)
        # sin(rho) = Re / Rs
        # Если alpha (надирный угол) > rho, то берем rho

        re = EarthConstants.RADIUS
        # Угол до горизонта от спутника
        rho_horizon = np.arcsin(re / active_sats_r)

        # Реальный theta_max (геоцентрический угол зоны покрытия)
        # Ограничиваем угол обзора горизонтом
        effective_alpha = np.minimum(alpha_rad, rho_horizon)

        # Формула: theta = arcsin( (Rs/Re)*sin(alpha) ) - alpha
        # Используем effective_alpha
        val = (active_sats_r / re) * np.sin(effective_alpha)
        # Защита от floating point errors (val > 1)
        val = np.clip(val, -1.0, 1.0)

        theta_max = np.arcsin(val) - effective_alpha
        min_cos_theta = np.cos(theta_max)  # Порог для скалярного произведения

        # Нормализованные векторы спутников
        active_sats_units = active_sats_pos / active_sats_r[:, np.newaxis]

        # ПРОВЕРКА ПОКРЫТИЯ (Матричное умножение)
        # Dot product matrix: (N_grid, N_active_sats) = Grid(N,3) @ Sats(M,3).T
        # Это может быть много памяти. Разбиваем на батчи если нужно.
        # Для 60x120 = 7200 точек и ~1000 спутников это 7.2 млн float -> 28MB RAM. OK.

        dots = grid_points @ active_sats_units.T

        # Проверяем условие dot >= min_cos_theta для каждого спутника
        # Broadcasting: dots (N_grid, N_sats) >= min_cos_theta (1, N_sats)
        is_covered_by_sat = dots >= min_cos_theta[np.newaxis, :]

        # Точка покрыта, если она видна ХОТЯ БЫ ОДНИМ спутником (any по оси спутников)
        point_covered = np.any(is_covered_by_sat, axis=1)

        covered_count = np.sum(point_covered)
        percent = covered_count / n_points * 100

        # Визуализация (ASCII Map)
        print("\nКарта покрытия (Меркатор):")
        map_grid = point_covered.reshape(grid_res, grid_res * 2)
        for row in map_grid:
            print("".join(["█" if x else "·" for x in row]))

        print("\n" + "=" * 60)
        print(f"Покрытие: {percent:.2f}%")
        print(f"Необслуживаемые точки: {n_points - covered_count}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--constellation', type=str, default='Starlink')
    parser.add_argument('--elevation', type=float, default=25.0)
    parser.add_argument('--alpha', type=float, default=40.0)
    parser.add_argument('--grid', type=int, default=40)
    parser.add_argument('--epoch', type=float, default=0.0)
    parser.add_argument('--use-j2', action='store_true')
    args = parser.parse_args()

    solver = Task4Solver()
    solver.solve(args.constellation, args.elevation, args.alpha,
                 args.grid, args.epoch, args.use_j2)


if __name__ == '__main__':
    main()
