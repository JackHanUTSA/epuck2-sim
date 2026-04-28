from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
from typing import Iterable


@dataclass(frozen=True)
class RobotConfig:
    name: str
    ip: str


@dataclass(frozen=True)
class TeamInterfaceConfig:
    robots: list[RobotConfig] = field(default_factory=list)
    camera_source: str = ''
    startup_command: str = ''
    webots_command: str = ''


@dataclass(frozen=True)
class TeamCommand:
    linear_x: float
    angular_z: float
    label: str


@dataclass(frozen=True)
class TeamConnectionSummary:
    connected_count: int
    total_count: int
    team_ready: bool
    details: dict[str, bool]


def _is_valid_ipv4(ip_text: str) -> bool:
    try:
        ipaddress.IPv4Address(ip_text)
        return True
    except ipaddress.AddressValueError:
        return False


def validate_robot_configs(config: TeamInterfaceConfig) -> list[str]:
    errors: list[str] = []
    seen_ips: set[str] = set()
    for index, robot in enumerate(config.robots, start=1):
        name = robot.name.strip()
        ip = robot.ip.strip()
        if not name:
            errors.append(f'Robot {index} name is required.')
        if not ip:
            errors.append(f'Robot {index} IP is required.')
            continue
        if not _is_valid_ipv4(ip):
            errors.append(f'Robot {name or index} must use a valid IPv4 address.')
            continue
        if ip in seen_ips:
            errors.append(f'Robot {name or index} has a duplicate IP address: {ip}.')
        seen_ips.add(ip)
    return errors


def build_team_command(order: str, *, linear_speed: float, angular_speed: float) -> TeamCommand:
    normalized = order.strip().lower()
    mapping = {
        'forward': TeamCommand(linear_x=linear_speed, angular_z=0.0, label='forward'),
        'backward': TeamCommand(linear_x=-linear_speed, angular_z=0.0, label='backward'),
        'left': TeamCommand(linear_x=0.0, angular_z=angular_speed, label='left'),
        'right': TeamCommand(linear_x=0.0, angular_z=-angular_speed, label='right'),
        'stop': TeamCommand(linear_x=0.0, angular_z=0.0, label='stop'),
        'spin_left': TeamCommand(linear_x=linear_speed * 0.5, angular_z=angular_speed, label='spin_left'),
        'spin_right': TeamCommand(linear_x=linear_speed * 0.5, angular_z=-angular_speed, label='spin_right'),
    }
    if normalized not in mapping:
        raise ValueError(f'Unsupported team order: {order}')
    return mapping[normalized]


def summarize_connection_results(results: dict[str, bool] | Iterable[tuple[str, bool]]) -> TeamConnectionSummary:
    details = dict(results)
    total_count = len(details)
    connected_count = sum(1 for ok in details.values() if ok)
    return TeamConnectionSummary(
        connected_count=connected_count,
        total_count=total_count,
        team_ready=total_count > 0 and connected_count == total_count,
        details=details,
    )
