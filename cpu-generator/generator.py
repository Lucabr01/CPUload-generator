from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Final

import numpy as np


@dataclass(frozen=True, slots=True)
class SimulationTime:
    """Target time used for load generation."""

    elapsed_seconds: int
    day_index: int
    seconds_of_day: int

    hour: int
    minute: int
    second: int

    stats_bin_index: int


class CPULoadGenerator:
    """
    Generate CPU loads synchronized with Sinergym time.

    Load names must match the Sinergym context keys. Call reset() at the
    start of each episode.
    """

    SECONDS_PER_DAY: Final[int] = 24 * 60 * 60
    HALF_HOUR_SECONDS: Final[int] = 30 * 60
    HOUR_SECONDS: Final[int] = 60 * 60
    DAYS_PER_WEEK: Final[int] = 7
    WEEKDAY_NAMES: Final[dict[str, int]] = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    EXPECTED_STATS_BINS: Final[int] = 48
    EXPECTED_WEEKLY_STATS_BINS: Final[int] = 7 * 24
    SAMPLE_TYPES: Final[frozenset[str]] = frozenset(
        {"whitenoise", "correlation", "correlation-week"}
    )
    DEFAULT_PHI_MIN: Final[float] = 0.7
    DEFAULT_PHI_MAX: Final[float] = 0.95
    STATS_FILE: Final[Path] = (
        Path(__file__).resolve().parent
        / "dataset"
        / "stats_daily.csv"
    )
    WEEKLY_STATS_FILE: Final[Path] = (
        Path(__file__).resolve().parent
        / "dataset"
        / "stats_weekly.csv"
    )

    def __init__(
        self,
        load_names: Sequence[str],
        step_seconds: int,
        stats_bin_seconds: int | None = None,
        seed: int | None = None,
        cached: bool = True,
        sample_type: str = "whitenoise",
        phi_min: float | None = None,
        phi_max: float | None = None,
    ) -> None:
        """
        Args:
            load_names: Sinergym context keys to generate.
            step_seconds: Sinergym step duration.
            stats_bin_seconds: Statistics interval; inferred when None.
            seed: Optional reproducibility seed.
            cached: Reuse one sample within each statistics interval.
            sample_type: ``whitenoise``, ``correlation``, or
                ``correlation-week``.
            phi_min: Optional minimum AR(1) coefficient. Valid only for
                correlated sample types.
            phi_max: Optional maximum AR(1) coefficient, strictly below
                one. Valid only for correlated sample types.
        """
        normalized_names = tuple(
            str(name).strip()
            for name in load_names
        )

        if not normalized_names:
            raise ValueError(
                "At least one load name must be specified."
            )

        if any(not name for name in normalized_names):
            raise ValueError(
                "Load names cannot be empty."
            )

        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError(
                "Load names must be unique."
            )

        if not isinstance(sample_type, str):
            raise TypeError(
                "sample_type must be a string."
            )

        normalized_sample_type = sample_type.strip().lower()

        if normalized_sample_type not in self.SAMPLE_TYPES:
            raise ValueError(
                "sample_type must be one of "
                f"{sorted(self.SAMPLE_TYPES)}, received "
                f'"{sample_type}".'
            )

        expected_stats_bin_seconds = (
            self.HOUR_SECONDS
            if normalized_sample_type == "correlation-week"
            else self.HALF_HOUR_SECONDS
        )

        if stats_bin_seconds is None:
            normalized_stats_bin_seconds = (
                expected_stats_bin_seconds
            )
        else:
            if isinstance(stats_bin_seconds, bool) or not isinstance(
                stats_bin_seconds,
                int,
            ):
                raise TypeError(
                    "stats_bin_seconds must be an integer or None."
                )
            normalized_stats_bin_seconds = stats_bin_seconds

        if normalized_stats_bin_seconds <= 0:
            raise ValueError(
                "stats_bin_seconds must be greater than zero."
            )

        if normalized_stats_bin_seconds != expected_stats_bin_seconds:
            raise ValueError(
                f'sample_type="{normalized_sample_type}" requires '
                "stats_bin_seconds to be "
                f"{expected_stats_bin_seconds}."
            )

        if isinstance(step_seconds, bool) or not isinstance(
            step_seconds,
            int,
        ):
            raise TypeError(
                "step_seconds must be an integer."
            )

        if step_seconds <= 0:
            raise ValueError(
                "step_seconds must be greater than zero."
            )

        if expected_stats_bin_seconds % step_seconds != 0:
            raise ValueError(
                "step_seconds must divide exactly the statistics "
                f"bin duration ({expected_stats_bin_seconds} seconds)."
            )

        if not isinstance(cached, bool):
            raise TypeError(
                "cached must be a boolean."
            )

        if (
            normalized_sample_type == "whitenoise"
            and (phi_min is not None or phi_max is not None)
        ):
            raise ValueError(
                "phi_min and phi_max can only be set when "
                'sample_type is "correlation" or '
                '"correlation-week".'
            )

        effective_phi_min = (
            self.DEFAULT_PHI_MIN
            if phi_min is None
            else phi_min
        )
        effective_phi_max = (
            self.DEFAULT_PHI_MAX
            if phi_max is None
            else phi_max
        )

        if (
            isinstance(effective_phi_min, (bool, str, bytes))
            or isinstance(effective_phi_max, (bool, str, bytes))
        ):
            raise TypeError(
                "phi_min and phi_max must be numeric."
            )

        try:
            normalized_phi_min = float(effective_phi_min)
            normalized_phi_max = float(effective_phi_max)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "phi_min and phi_max must be numeric."
            ) from exc

        if not (
            isfinite(normalized_phi_min)
            and isfinite(normalized_phi_max)
        ):
            raise ValueError(
                "phi_min and phi_max must be finite."
            )

        if not (
            0.0
            <= normalized_phi_min
            <= normalized_phi_max
            < 1.0
        ):
            raise ValueError(
                "phi bounds must satisfy "
                "0 <= phi_min <= phi_max < 1."
            )

        self.load_names = normalized_names
        self.step_seconds = step_seconds
        self.stats_bin_seconds = normalized_stats_bin_seconds
        self.cached = cached
        self.sample_type = normalized_sample_type
        self.phi_min = normalized_phi_min
        self.phi_max = normalized_phi_max

        self.n_stats_bins = (
            self.SECONDS_PER_DAY
            // self.stats_bin_seconds
        )

        (
            self._stats_means,
            self._stats_stds,
        ) = self._load_statistics()

        self.rng = np.random.default_rng(seed)

        # Per-episode state.
        self._episode_step: int
        self._last_target_time: SimulationTime | None
        self._last_values: dict[str, float] | None
        self._last_stats_bin_key: tuple[int, int] | None
        self._last_bin_values: dict[str, float] | None
        self._ar_phi: float | None
        self._ar_residuals: np.ndarray | None
        self._ar_current_index: int | None
        self._ar_last_elapsed_seconds: int | None
        self._clipping_count: int
        self._episode_origin_seconds: int

        self.reset()

    @property
    def n_loads(self) -> int:
        """Number of loads generated per call."""
        return len(self.load_names)

    @property
    def episode_step(self) -> int:
        """Number of samples generated in the current episode."""
        return self._episode_step

    @property
    def last_target_time(self) -> SimulationTime | None:
        """Most recent target time."""
        return self._last_target_time

    @property
    def last_values(self) -> dict[str, float] | None:
        """Return a copy of the most recently generated loads."""
        if self._last_values is None:
            return None

        return self._last_values.copy()

    @property
    def phi(self) -> float | None:
        """Trace AR(1) coefficient, or None for white noise."""
        return self._ar_phi

    @property
    def clipping_count(self) -> int:
        """Number of values clipped to [0, 1] in this episode."""
        return self._clipping_count

    @property
    def episode_start_time(self) -> SimulationTime:
        """Calendar-aligned start time for the current episode."""
        return self._simulation_time_from_seconds(
            self._episode_origin_seconds
        )

    def reset(
        self,
        seed: int | None = None,
        *,
        start_weekday: int = 0,
        start_hour: int = 0,
        start_minute: int = 0,
        start_second: int = 0,
    ) -> None:
        """
        Reset state for a new episode.

        Args:
            seed: Restart the random stream when provided; otherwise
                continue it. Correlated modes always draw new AR state.
            start_weekday: Episode start day, where Monday is zero.
            start_hour: Episode start hour.
            start_minute: Episode start minute.
            start_second: Episode start second.
        """
        time_components = {
            "start_weekday": (start_weekday, 0, 6),
            "start_hour": (start_hour, 0, 23),
            "start_minute": (start_minute, 0, 59),
            "start_second": (start_second, 0, 59),
        }

        for name, (value, minimum, maximum) in time_components.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{name} must belong to [{minimum}, {maximum}]."
                )

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._episode_origin_seconds = (
            start_weekday * self.SECONDS_PER_DAY
            + start_hour * self.HOUR_SECONDS
            + start_minute * 60
            + start_second
        )

        self._episode_step = 0
        self._last_target_time = None
        self._last_values = None
        self._last_stats_bin_key = None
        self._last_bin_values = None
        self._clipping_count = 0

        if self.sample_type in {"correlation", "correlation-week"}:
            self._ar_phi = float(
                self.rng.uniform(
                    self.phi_min,
                    self.phi_max,
                )
            )
            self._ar_residuals = np.asarray(
                self.rng.normal(
                    loc=0.0,
                    scale=1.0,
                    size=self.n_loads,
                ),
                dtype=float,
            )
            if self.cached:
                origin_day, origin_seconds_of_day = divmod(
                    self._episode_origin_seconds,
                    self.SECONDS_PER_DAY,
                )
                self._ar_current_index = (
                    origin_day * self.n_stats_bins
                    + origin_seconds_of_day // self.stats_bin_seconds
                )
            else:
                self._ar_current_index = 0
            self._ar_last_elapsed_seconds = (
                self._episode_origin_seconds
            )
        else:
            self._ar_phi = None
            self._ar_residuals = None
            self._ar_current_index = None
            self._ar_last_elapsed_seconds = None

    def reset_from_sinergym(
        self,
        env: object,
        seed: int | None = None,
    ) -> None:
        """Reset using the start date and time exposed by Sinergym."""
        unwrapped_env = getattr(env, "unwrapped", env)
        runperiod = getattr(unwrapped_env, "runperiod", None)

        if not isinstance(runperiod, Mapping):
            raise TypeError(
                "Sinergym env.runperiod must be a mapping."
            )

        if "start_weekday" not in runperiod:
            raise ValueError(
                'Sinergym env.runperiod has no "start_weekday".'
            )

        start_weekday = runperiod["start_weekday"]

        if isinstance(start_weekday, str):
            normalized_weekday = self.WEEKDAY_NAMES.get(
                start_weekday.strip().lower()
            )
            if normalized_weekday is None:
                raise ValueError(
                    "Unknown Sinergym start weekday: "
                    f'"{start_weekday}".'
                )
            start_weekday = normalized_weekday

        self.reset(
            seed=seed,
            start_weekday=start_weekday,
            start_hour=runperiod.get("start_hour", 0),
            start_minute=runperiod.get("start_minute", 0),
            start_second=runperiod.get("start_second", 0),
        )

    def _load_statistics(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """
        Load and validate half-hour statistics.

        ``HH.MM`` labels are times, not decimal hours.
        """
        if self.sample_type == "correlation-week":
            return self._load_weekly_statistics()

        if not self.STATS_FILE.is_file():
            raise FileNotFoundError(
                "Half-hour statistics file not found: "
                f"{self.STATS_FILE}"
            )

        required_columns = {
            "half_hour_interval",
            "mean_global_load",
        }

        with self.STATS_FILE.open(
            mode="r",
            encoding="utf-8-sig",
            newline="",
        ) as stats_file:
            reader = csv.DictReader(stats_file)
            fieldnames = set(reader.fieldnames or ())

            missing_columns = required_columns - fieldnames

            if missing_columns:
                raise ValueError(
                    "Missing columns in half-hour statistics: "
                    f"{sorted(missing_columns)}"
                )

            if not (
                "std_global_load" in fieldnames
                or "variance_global_load" in fieldnames
            ):
                raise ValueError(
                    "Half-hour statistics must contain "
                    "std_global_load or variance_global_load."
                )

            rows = list(reader)

        if len(rows) != self.EXPECTED_STATS_BINS:
            raise ValueError(
                "Half-hour statistics must contain exactly "
                f"{self.EXPECTED_STATS_BINS} rows, received "
                f"{len(rows)}."
            )

        means: list[float] = []
        stds: list[float] = []

        for index, row in enumerate(rows):
            seconds_from_midnight = (
                index * self.stats_bin_seconds
            )
            hour, remaining_seconds = divmod(
                seconds_from_midnight,
                3600,
            )
            minute = remaining_seconds // 60
            expected_interval = f"{hour:02d}.{minute:02d}"
            received_interval = (
                row["half_hour_interval"] or ""
            ).strip()
            normalized_interval = received_interval.replace(
                ":",
                ".",
            )

            if normalized_interval != expected_interval:
                raise ValueError(
                    "Half-hour statistics are not ordered from "
                    "00.00 to 23.30: expected "
                    f'"{expected_interval}" at row {index}, '
                    f'received "{received_interval}".'
                )

            try:
                mean = float(row["mean_global_load"])
                std = (
                    float(row["std_global_load"])
                    if "std_global_load" in row
                    else None
                )
                variance = (
                    float(row["variance_global_load"])
                    if "variance_global_load" in row
                    else None
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Mean, standard deviation, and variance must "
                    "be numeric "
                    f"at statistics row {index}."
                ) from exc

            if not isfinite(mean):
                raise ValueError(
                    "Mean must be finite at statistics row "
                    f"{index}, received {mean}."
                )

            if std is not None and not isfinite(std):
                raise ValueError(
                    "Standard deviation must be finite at "
                    f"statistics row {index}, received {std}."
                )

            if std is not None and std < 0:
                raise ValueError(
                    "Standard deviation cannot be negative at "
                    f"statistics row {index}, received {std}."
                )

            if variance is not None and not isfinite(variance):
                raise ValueError(
                    "Variance must be finite at statistics row "
                    f"{index}, received {variance}."
                )

            if variance is not None and variance < 0:
                raise ValueError(
                    "Variance cannot be negative at statistics row "
                    f"{index}, received {variance}."
                )

            if std is None:
                assert variance is not None
                std = sqrt(variance)
            elif variance is not None and not np.isclose(
                std * std,
                variance,
                rtol=1e-7,
                atol=1e-12,
            ):
                raise ValueError(
                    "Standard deviation and variance are "
                    "inconsistent at statistics row "
                    f"{index}: std^2={std * std}, "
                    f"variance={variance}."
                )

            means.append(mean)
            stds.append(std)

        return tuple(means), tuple(stds)

    def _load_weekly_statistics(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Load and validate 168 hourly weekly statistics."""
        if not self.WEEKLY_STATS_FILE.is_file():
            raise FileNotFoundError(
                "Weekly statistics file not found: "
                f"{self.WEEKLY_STATS_FILE}"
            )

        required_columns = {
            "day",
            "hour",
            "mean_global_load",
        }

        with self.WEEKLY_STATS_FILE.open(
            mode="r",
            encoding="utf-8-sig",
            newline="",
        ) as stats_file:
            reader = csv.DictReader(stats_file)
            fieldnames = set(reader.fieldnames or ())
            missing_columns = required_columns - fieldnames

            if missing_columns:
                raise ValueError(
                    "Missing columns in weekly statistics: "
                    f"{sorted(missing_columns)}"
                )

            if not (
                "std_global_load" in fieldnames
                or "variance_global_load" in fieldnames
            ):
                raise ValueError(
                    "Weekly statistics must contain "
                    "std_global_load or variance_global_load."
                )

            rows = list(reader)

        if len(rows) != self.EXPECTED_WEEKLY_STATS_BINS:
            raise ValueError(
                "Weekly statistics must contain exactly "
                f"{self.EXPECTED_WEEKLY_STATS_BINS} rows, "
                f"received {len(rows)}."
            )

        means: list[float] = []
        stds: list[float] = []

        for index, row in enumerate(rows):
            expected_day = index // 24 + 1
            expected_hour = f"{index % 24:02d}:00"

            try:
                received_day = int(row["day"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Weekly day must be an integer at statistics "
                    f"row {index}."
                ) from exc

            received_hour = (
                row["hour"] or ""
            ).strip().replace(".", ":")

            if (
                received_day != expected_day
                or received_hour != expected_hour
            ):
                raise ValueError(
                    "Weekly statistics are not ordered by day "
                    "1-7 and hour 00:00-23:00: expected "
                    f'day {expected_day}, hour "{expected_hour}" '
                    f"at row {index}, received day "
                    f'{received_day}, hour "{received_hour}".'
                )

            try:
                mean = float(row["mean_global_load"])
                std = (
                    float(row["std_global_load"])
                    if "std_global_load" in row
                    else None
                )
                variance = (
                    float(row["variance_global_load"])
                    if "variance_global_load" in row
                    else None
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Weekly mean, standard deviation, and variance "
                    "must be numeric at statistics row "
                    f"{index}."
                ) from exc

            if not isfinite(mean):
                raise ValueError(
                    "Weekly mean must be finite at statistics row "
                    f"{index}, received {mean}."
                )

            if std is not None and (
                not isfinite(std) or std < 0
            ):
                raise ValueError(
                    "Weekly standard deviation must be finite and "
                    "non-negative at statistics row "
                    f"{index}, received {std}."
                )

            if variance is not None and (
                not isfinite(variance) or variance < 0
            ):
                raise ValueError(
                    "Weekly variance must be finite and non-negative "
                    f"at statistics row {index}, received "
                    f"{variance}."
                )

            if std is None:
                assert variance is not None
                std = sqrt(variance)
            elif variance is not None and not np.isclose(
                std * std,
                variance,
                rtol=1e-7,
                atol=1e-12,
            ):
                raise ValueError(
                    "Weekly standard deviation and variance are "
                    "inconsistent at statistics row "
                    f"{index}: std^2={std * std}, "
                    f"variance={variance}."
                )

            means.append(mean)
            stds.append(std)

        return tuple(means), tuple(stds)

    def next(
        self,
        target_elapsed_hours: float,
    ) -> dict[str, float]:
        """
        Generate loads for an explicit target time.

        Args:
            target_elapsed_hours: Target time relative to the episode
                start. When the value will be applied by the next
                ``env.step()``, pass the next timestep time rather than
                the currently observed Sinergym time. For fixed-step
                environments, prefer :meth:`next_step`.

        Returns:
            Values for ``env.update_context()``.

        Raises:
            RuntimeError: If time is not strictly increasing.
        """
        target_time = self.get_target_time(
            target_elapsed_hours
        )

        if (
            self._last_target_time is not None
            and target_time.elapsed_seconds
            <= self._last_target_time.elapsed_seconds
        ):
            raise RuntimeError(
                "Simulation time is not strictly increasing. "
                "The environment may have started a new episode "
                "without calling generator.reset(), or next() may "
                "have been called more than once for the same "
                "Sinergym observation."
            )

        generated_values = self.sample_at(
            target_time
        )

        validated_values = self._validate_sample(
            generated_values
        )

        self._last_target_time = target_time
        self._last_values = validated_values.copy()
        self._episode_step += 1

        return validated_values

    def next_step(self) -> dict[str, float]:
        """Generate the value for the next fixed simulation timestep.

        The initial context belongs to ``t0``. Therefore, immediately
        after :meth:`reset`, the first call targets ``t1`` at
        ``step_seconds``; the second targets ``t2``, and so on. Integer
        step arithmetic is used before converting to hours, preventing
        irregular Sinergym telemetry from changing the sampling grid.

        Do not mix this method with manual calls to :meth:`next` within
        the same episode unless those calls follow the identical grid.
        """
        target_elapsed_seconds = (
            self._episode_step + 1
        ) * self.step_seconds
        return self.next(
            target_elapsed_seconds / self.HOUR_SECONDS
        )

    def get_target_time(
        self,
        target_elapsed_hours: float,
    ) -> SimulationTime:
        """
        Return the observed EnergyPlus time aligned to the episode.

        Rounding to integer seconds prevents floating-point bin errors.
        """
        target_elapsed_hours = float(
            target_elapsed_hours
        )

        if not isfinite(target_elapsed_hours):
            raise ValueError(
                "target_elapsed_hours must be finite."
            )

        if target_elapsed_hours < 0:
            raise ValueError(
                "target_elapsed_hours cannot be negative."
            )

        current_elapsed_seconds = int(
            round(target_elapsed_hours * 3600)
        )

        target_elapsed_seconds = (
            self._episode_origin_seconds
            + current_elapsed_seconds
        )

        return self._simulation_time_from_seconds(
            target_elapsed_seconds
        )

    def _simulation_time_from_seconds(
        self,
        elapsed_seconds: int,
    ) -> SimulationTime:
        """Build a calendar-aligned simulation timestamp."""

        day_index, seconds_of_day = divmod(
            elapsed_seconds,
            self.SECONDS_PER_DAY,
        )

        hour, remaining_seconds = divmod(
            seconds_of_day,
            3600,
        )

        minute, second = divmod(
            remaining_seconds,
            60,
        )

        stats_bin_index = (
            seconds_of_day
            // self.stats_bin_seconds
        )

        return SimulationTime(
            elapsed_seconds=elapsed_seconds,
            day_index=day_index,
            seconds_of_day=seconds_of_day,
            hour=hour,
            minute=minute,
            second=second,
            stats_bin_index=stats_bin_index,
        )

    def sample_at(
        self,
        target_time: SimulationTime,
    ) -> dict[str, float]:
        """
        Generate clipped CPU loads for a target time.

        Cached values are constant within a statistics interval. Uncached
        AR modes scale phi using the elapsed time between observations.
        All loads currently share the same global statistics.
        """
        if not isinstance(target_time, SimulationTime):
            raise TypeError(
                "target_time must be a SimulationTime instance."
            )

        stats_bin_index = target_time.stats_bin_index

        if not 0 <= stats_bin_index < self.n_stats_bins:
            raise ValueError(
                "target_time.stats_bin_index must belong to "
                f"[0, {self.n_stats_bins - 1}], received "
                f"{stats_bin_index}."
            )

        stats_bin_key = (
            target_time.day_index,
            stats_bin_index,
        )

        if (
            self.cached
            and stats_bin_key == self._last_stats_bin_key
            and self._last_bin_values is not None
        ):
            return self._last_bin_values.copy()

        statistics_index = self._statistics_index(target_time)
        mean = self._stats_means[statistics_index]
        std = self._stats_stds[statistics_index]

        if self.sample_type == "whitenoise":
            samples = np.asarray(
                self.rng.normal(
                    loc=mean,
                    scale=std,
                    size=self.n_loads,
                ),
                dtype=float,
            )
        else:
            samples = self._sample_correlated(
                target_time=target_time,
                mean=mean,
                std=std,
            )

        if not np.all(np.isfinite(samples)):
            raise ValueError(
                "The random generator produced a non-finite load."
            )

        clipped_samples = np.clip(samples, 0.0, 1.0)
        self._clipping_count += int(
            np.count_nonzero(samples != clipped_samples)
        )
        generated_values = {
            load_name: float(value)
            for load_name, value in zip(
                self.load_names,
                clipped_samples,
                strict=True,
            )
        }

        if self.cached:
            self._last_stats_bin_key = stats_bin_key
            self._last_bin_values = generated_values.copy()

        return generated_values

    def _statistics_index(
        self,
        target_time: SimulationTime,
    ) -> int:
        """Return the daily or weekly statistics index."""
        if self.sample_type == "correlation-week":
            week_day_index = (
                target_time.day_index % self.DAYS_PER_WEEK
            )
            return (
                week_day_index * self.n_stats_bins
                + target_time.stats_bin_index
            )

        return target_time.stats_bin_index

    def _sample_correlated(
        self,
        target_time: SimulationTime,
        mean: float,
        std: float,
    ) -> np.ndarray:
        """Generate AR(1) samples at the next bin or step."""
        if (
            self._ar_phi is None
            or self._ar_residuals is None
            or self._ar_current_index is None
            or self._ar_last_elapsed_seconds is None
        ):
            raise RuntimeError(
                "AR(1) state is not initialized. Call reset() "
                "before generating correlated samples."
            )

        if self.cached:
            target_index = (
                target_time.day_index * self.n_stats_bins
                + target_time.stats_bin_index
            )
            effective_phi = self._ar_phi
            steps_to_advance = target_index - self._ar_current_index

            if steps_to_advance < 0:
                raise RuntimeError(
                    "Correlated sampling time cannot move backwards."
                )

            innovation_scale = sqrt(
                1.0 - effective_phi * effective_phi
            )

            for _ in range(steps_to_advance):
                innovations = np.asarray(
                    self.rng.normal(
                        loc=0.0,
                        scale=1.0,
                        size=self.n_loads,
                    ),
                    dtype=float,
                )
                self._ar_residuals = (
                    effective_phi * self._ar_residuals
                    + innovation_scale * innovations
                )

            self._ar_current_index = target_index
        else:
            elapsed_delta = (
                target_time.elapsed_seconds
                - self._ar_last_elapsed_seconds
            )

            if elapsed_delta < 0:
                raise RuntimeError(
                    "Correlated sampling time cannot move backwards."
                )

            if elapsed_delta > 0:
                effective_phi = self._ar_phi ** (
                    elapsed_delta / self.stats_bin_seconds
                )
                innovation_scale = sqrt(
                    1.0 - effective_phi * effective_phi
                )
                innovations = np.asarray(
                    self.rng.normal(
                        loc=0.0,
                        scale=1.0,
                        size=self.n_loads,
                    ),
                    dtype=float,
                )
                self._ar_residuals = (
                    effective_phi * self._ar_residuals
                    + innovation_scale * innovations
                )
                self._ar_current_index += 1

            self._ar_last_elapsed_seconds = (
                target_time.elapsed_seconds
            )

        return np.asarray(
            mean + std * self._ar_residuals,
            dtype=float,
        )

    def _validate_sample(
        self,
        values: dict[str, float],
    ) -> dict[str, float]:
        """Validate sample_at() output for Sinergym."""
        if not isinstance(values, dict):
            raise TypeError(
                "sample_at() must return a dictionary. "
                
            )

        expected_names = set(self.load_names)
        received_names = set(values)

        missing_names = (
            expected_names - received_names
        )

        unexpected_names = (
            received_names - expected_names
        )

        if missing_names:
            raise ValueError(
                "Missing generated loads: "
                f"{sorted(missing_names)}"
            )

        if unexpected_names:
            raise ValueError(
                "Unexpected generated loads: "
                f"{sorted(unexpected_names)}"
            )

        normalized_values: dict[str, float] = {}

        for load_name in self.load_names:
            try:
                value = float(values[load_name])
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f'Load "{load_name}" must be numeric.'
                ) from exc

            if not isfinite(value):
                raise ValueError(
                    f'Load "{load_name}" is not finite: '
                    f"{value}."
                )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f'Load "{load_name}" must belong to '
                    f"[0, 1], received {value}."
                )

            normalized_values[load_name] = value

        return normalized_values
