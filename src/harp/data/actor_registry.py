"""Concrete ActorRegistry implementation for the SMIM framework.

This module provides a fully-implemented ActorRegistry with:
- O(1) actor lookup via an internal index dict
- Layer and type filtering
- ``from_taxonomy(path)`` class method that parses a YAML block
  embedded in a taxonomy Markdown file (see docs/smim/actor_taxonomy.md)

Usage::

    from harp.data.actor_registry import ActorRegistry
    registry = ActorRegistry.from_taxonomy("docs/smim/actor_taxonomy.md")
    fed = registry.index_of("fed_us")
    upstream = registry.actors_in_layer(Layer.UPSTREAM)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from harp.interfaces import Actor, ActorType, Layer


@dataclass
class ActorRegistry:
    """Concrete registry of all actors in the SMIM system.

    Provides O(1) lookup by actor_id via an internal index built at
    construction time.

    Attributes:
        actors: Ordered list of Actor instances.
    """

    actors: list[Actor]
    _index: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {actor.actor_id: i for i, actor in enumerate(self.actors)}

    @property
    def N(self) -> int:
        """Total number of actors."""
        return len(self.actors)

    def index_of(self, actor_id: str) -> int:
        """Return the integer index for actor_id.

        Args:
            actor_id: Unique actor identifier string.

        Returns:
            Zero-based index of the actor in ``self.actors``.

        Raises:
            KeyError: If actor_id is not in the registry.
        """
        try:
            return self._index[actor_id]
        except KeyError:
            raise KeyError(f"Actor {actor_id!r} not found in registry") from None

    def actors_in_layer(self, layer: Layer) -> list[Actor]:
        """Return all actors assigned to the given layer.

        Args:
            layer: One of Layer.EXOGENOUS, UPSTREAM, TRANSMISSION, DOWNSTREAM.

        Returns:
            List of Actor instances with ``actor.layer == layer``.
        """
        return [a for a in self.actors if a.layer == layer]

    def actors_of_type(self, actor_type: ActorType) -> list[Actor]:
        """Return all actors with the given type.

        Args:
            actor_type: ActorType enum value.

        Returns:
            List of Actor instances with ``actor.actor_type == actor_type``.
        """
        return [a for a in self.actors if a.actor_type == actor_type]

    @classmethod
    def from_taxonomy(cls, path: str | Path) -> ActorRegistry:
        """Build a registry from the exemplar actors in a taxonomy Markdown file.

        The Markdown file must contain exactly one fenced YAML code block
        (delimited by `` ```yaml `` … `` ``` ``) whose content is a mapping
        with an ``actors`` key listing actor dicts.  The block in
        ``docs/smim/actor_taxonomy.md`` serves as the canonical example.

        Args:
            path: Path to the taxonomy Markdown file.

        Returns:
            ActorRegistry populated from the embedded YAML actor data.

        Raises:
            FileNotFoundError: If the Markdown file does not exist.
            ValueError: If no YAML code block is found in the file.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")

        # Find the first fenced ```yaml … ``` block
        match = re.search(r"```yaml\n(.*?)```", text, re.DOTALL)
        if not match:
            raise ValueError(
                f"No fenced YAML code block found in {path}. "
                "The taxonomy file must contain a block delimited by ```yaml / ```."
            )

        data = yaml.safe_load(match.group(1))
        if not data or "actors" not in data:
            raise ValueError(
                f"YAML block in {path} must have a top-level 'actors' key."
            )

        actors = [
            Actor(
                actor_id=a["actor_id"],
                name=a["name"],
                actor_type=ActorType(a["actor_type"]),
                layer=Layer(int(a["layer"])),
                geography=a["geography"],
                sector=a["sector"],
                external_ids=a.get("external_ids", {}),
            )
            for a in data["actors"]
        ]
        return cls(actors=actors)

    @classmethod
    def from_json(cls, path: str | Path) -> ActorRegistry:
        """Load a registry from a JSON file produced by ``to_json``.

        Args:
            path: Path to a JSON file with a top-level ``actors`` key.

        Returns:
            ActorRegistry populated from the JSON actor list.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON has no ``actors`` key.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if "actors" not in data:
            raise ValueError(f"JSON file {path} must have a top-level 'actors' key.")
        actors = [
            Actor(
                actor_id=a["actor_id"],
                name=a["name"],
                actor_type=ActorType(a["actor_type"]),
                layer=Layer(int(a["layer"])),
                geography=a["geography"],
                sector=a["sector"],
                external_ids=a.get("external_ids", {}),
            )
            for a in data["actors"]
        ]
        return cls(actors=actors)

    def to_json(self, path: str | Path, **metadata: object) -> None:
        """Serialise registry to a JSON file.

        Args:
            path: Destination file path.
            **metadata: Extra top-level keys stored alongside ``actors``
                (e.g. ``universe_id="US-LC"``, ``description="..."``)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = dict(metadata)
        payload["actors"] = [
            {
                "actor_id": a.actor_id,
                "name": a.name,
                "actor_type": a.actor_type.value,
                "layer": a.layer.value,
                "geography": a.geography,
                "sector": a.sector,
                "external_ids": a.external_ids,
            }
            for a in self.actors
        ]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
