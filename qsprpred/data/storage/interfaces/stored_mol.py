from abc import ABC, abstractmethod
from typing import Any, Optional

from rdkit import Chem


class StoredMol(ABC):
    """A simple interface for a molecule that can be stored in a `ChemStore`.
    Molecules in the `ChemStore` have properties, representations, and
    can also have a parent molecule. Representations can be for example
    conformers, tautomers, or protomers of the parent molecule. Representations
    can also be used to encode docked poses with metadata attached as properties.
    """

    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({self.id}, {self.smiles})"

    def __eq__(self, other: "StoredMol") -> bool:
        return self.id == other.id

    @property
    @abstractmethod
    def origin(self) -> str:
        """Get the name of the storage where the molecule resides.

        Returns:
            str: The name of the storage.
        """

    @property
    @abstractmethod
    def parent(self) -> Optional["StoredMol"]:
        """Get the parent molecule of this representation.

        Returns:
            The parent molecule of this representation as a `StoredMol` instance.
        """

    @parent.setter
    @abstractmethod
    def parent(self, parent: Optional["StoredMol"]):
        """Set the parent molecule of this representation.

        Args:
            parent (StoredMol): The parent molecule of this representation.
        """

    @property
    @abstractmethod
    def smiles(self) -> str:
        """Get the SMILES of the molecule.

        Returns:
            str: The SMILES of the molecule.
        """

    @property
    @abstractmethod
    def id(self) -> str:
        """Get the identifier of the molecule.

        Returns:
            str: The identifier of the molecule.
        """

    @property
    @abstractmethod
    def props(self) -> dict[str, Any] | None:
        """Get the metadata of the molecule.

        Returns:
            dict: The metadata of the molecule.
        """

    @property
    @abstractmethod
    def representations(self) -> list["StoredMol"] | None:
        """Get the representations of the molecule.

        Returns:
            list: The representations of the molecule.
        """

    @representations.setter
    @abstractmethod
    def representations(self, representations: list["StoredMol"] | None):
        """Set the representations of the molecule.

        Args:
            representations (list): The representations of the molecule.
        """

    def as_rd_mol(self) -> Chem.Mol:
        """Get the RDKit molecule object of the standardized representation of this instance.

        Returns:
            `rdkit.Chem.Mol` instance
        """

        return Chem.MolFromSmiles(self.smiles)
