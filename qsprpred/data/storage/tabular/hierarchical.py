import os
import shutil
from typing import ClassVar, Literal, Iterable, Generator, Any, Sized

import pandas as pd
from rdkit import Chem

from qsprpred.data.chem.identifiers import ChemIdentifier
from qsprpred.data.chem.standardizers import ChemStandardizer
from qsprpred.data.processing.mol_processor import MolProcessor
from qsprpred.data.storage.interfaces.chem_store import ChemStore
from qsprpred.data.storage.interfaces.searchable import PropSearchable, SMARTSSearchable
from qsprpred.data.storage.interfaces.stored_mol import StoredMol
from qsprpred.data.storage.tabular.simple import PandasChemStore
from qsprpred.data.storage.tabular.stored_mol import TabularMol
from qsprpred.logs import logger
from qsprpred.utils.interfaces.summarizable import Summarizable
from qsprpred.utils.parallel import ParallelGenerator


class RepresentationMol(TabularMol):

    def as_rd_mol(self, add_props=False) -> Chem.Mol:
        sdf = self.props["sdf"]
        mol = Chem.MolFromMolBlock(
            sdf, strictParsing=False, sanitize=False, removeHs=False
        )
        if add_props:
            for prop in self.props:
                mol.SetProp(prop, str(self.props[prop]))
        return mol

    def sdf(self) -> str:
        return self.props["sdf"]

    def to_file(self, directory, extension=".csv") -> str:
        """
        Write a minimal file containing the SMILES and the ID of the molecule.
        Used for ligrep (.csv is the preferred format).
        """
        filename = os.path.join(directory, self.id + extension)
        if not os.path.isfile(filename):
            with open(filename, "w") as f:
                f.write("SMILES,id\n")
                f.write(f"{self.smiles},{self.id}\n")
        return filename


class PandasRepresentationStore(
    ChemStore,
    SMARTSSearchable,
    PropSearchable,
    Summarizable
):
    _notJSON: ClassVar = [*ChemStore._notJSON, "representations"]

    def __init__(
            self,
            name: str,
            path: str,
            chem_store: ChemStore | None = None,
            df: pd.DataFrame | None = None,
            store_format: str = "pkl",
            add_rdkit: bool = False,
            overwrite: bool = False,
            chunk_processor: ParallelGenerator = None,
            chunk_size: int | None = None,
            n_jobs: int = 1,
    ) -> None:
        super().__init__()
        self.storage = chem_store
        self.name = name
        self.path = path
        self.chunkProcessor = chunk_processor
        if df is not None:
            raise NotImplementedError(
                "Supplying an initial set of representations is not yet supported."
            )
        if overwrite:
            logger.warning(
                "Overwriting the representations will not clear the main storage."
                "Run clear() on the main storage to clear it separately."
            )
            self.clear()
        if not os.path.exists(self.metaFile):
            logger.info(f"Creating new representation store at {self.baseDir}")
            assert self.storage is not None, "Storage with molecules must be provided"
            self.representations = PandasChemStore(
                f"{self.name}_representations",
                path=self.baseDir,
                df=pd.DataFrame(
                    columns=[
                        self.idProp,
                        "parent_id",
                        self.smilesProp,
                        "sdf"
                    ]
                ),
                smiles_col=self.smilesProp,
                add_rdkit=add_rdkit,
                id_col=self.idProp,
                overwrite=overwrite,
                store_format=store_format,
                chunk_processor=chunk_processor,
                chunk_size=chunk_size,
                n_jobs=n_jobs,
            )
        else:
            logger.info(f"Loading representation store at {self.baseDir}")
            self.reload()

    @property
    def baseDir(self) -> str:
        return os.path.join(self.path, self.name)

    @property
    def metaFile(self) -> str:
        return os.path.join(self.baseDir, "meta.json")

    @property
    def smilesProp(self) -> str:
        return self.storage.smilesProp

    def getRepresentations(
            self,
            mol_id: str,
            recursive=True,
            is_root=False
    ) -> list[StoredMol]:
        """Find all representations of a molecule recursively.

        Args:
            mol_id (str):
                identifier of the molecule to find representations for
            recursive (bool):
                whether to find representations recursively or just one level
            is_root (bool):
                whether the molecule is the root molecule
                (the parent of all representations) -> will be searched for
                in the main storage
        """
        if not is_root:
            mol = self.representations.getMol(mol_id)
            mol.__class__ = RepresentationMol
        else:
            mol = self.storage.getMol(mol_id)
        children = list(self.representations.searchOnProperty(
            "parent_id",
            [mol_id],
            exact=True
        )) or None
        if children is not None:
            for child in children:
                child.__class__ = RepresentationMol
                child.parent = mol
                if recursive:
                    child.representations = self.getRepresentations(child.id)
        return children

    def getMol(self, mol_id: str) -> StoredMol:
        """Retrieve a molecule with all its representations attached.

        Args:
            mol_id (str):
                identifier of the molecule to retrieve

        Returns:
            (StoredMol):
                molecule with all its representations attached to its `representations` attribute
        """
        mol = self.storage.getMol(mol_id)
        children = self.getRepresentations(mol_id, is_root=True)
        mol.representations = children
        if mol.representations is not None:
            for rep in mol.representations:
                rep.parent = mol
        return mol

    def addMols(self, smiles: Iterable[str], props: dict[str, list] | None = None,
                *args, **kwargs) -> list[StoredMol]:
        """Add new representations to the store.

        It is required that
        the properties contain a 'parent_id' property that points to the
        parent molecule in the underlying `storage` object or another representation
        stored in this object itself.

        The 'sdf' property
        must also be provided, which defines the representation of the molecule
        in SDF format. Other properties can be provided as well to indicate the nature
        of the representation.

        Args:
            smiles:
                The SMILES of the representations to add.
            props:
                The properties of the representations to add.
            *args:
                Additional arguments.
            **kwargs:
                Additional keyword arguments.

        Returns:
            (list[StoredMol]):
                The added representations.
        """
        assert props is not None, "Properties must be provided."
        assert "parent_id" in props, \
            "Parent ID is missing. It must be provided as 'parent_id' property."
        assert "sdf" in props, \
            "SDF table is missing. It must be provided as 'sdf' property."
        # TODO: add checks for the parent_id and sdf properties
        return self.representations.addMols(smiles, props, *args, **kwargs)

    def removeRepresentations(self, mol_id: str):
        """Remove all representations of a molecule from the store."""
        reps = self.getRepresentations(mol_id, recursive=False, is_root=True)
        if reps is not None:
            for child in reps:
                self.removeRepresentations(child.id)
                self.representations.removeMol(child.id)

    def removeMol(self, mol_id: str):
        """Remove all representations of a molecule from the store."""
        return self.removeRepresentations(mol_id)

    def getMolIDs(self) -> tuple[str, ...]:
        """Get the identifiers of all representations in the store."""
        return self.storage.getMolIDs()

    def getMolCount(self):
        """Get the number of representations in the store."""
        return self.storage.getMolCount()

    def iterMols(self) -> Generator[StoredMol, None, None]:
        """Iterate over all molecules in the attached storage
        with their representations added.

        Yields:
            (StoredMol):
                molecule with all its representations attached to its `representations` attribute
        """
        for mol in self.storage.iterMols():
            reps = self.getRepresentations(mol.id, is_root=True)
            mol.representations = reps
            if mol.representations is not None:
                for rep in mol.representations:
                    rep.parent = mol
            yield mol

    def iterChunks(self, size: int | None = None, on_props: list | None = None,
                   chunk_type: Literal["mol", "smiles", "rdkit", "df"] = "mol") -> \
            Generator[list[StoredMol | str | Chem.Mol | pd.DataFrame], None, None]:
        """Iterate over chunks of molecules with their representations added.

        Args:
            size (int):
                size of the chunks to yield
            on_props (list):
                properties to chunk on
            chunk_type (str):
                type of the chunk to yield

        Yields:
            (list[StoredMol | str | Chem.Mol | pd.DataFrame]):
                chunk of molecules with all their representations attached to their `representations` attribute
        """
        for chunk in self.storage.iterChunks(size, on_props, chunk_type):
            if chunk_type == "mol":
                for mol in chunk:
                    reps = self.getRepresentations(mol.id, is_root=True)
                    mol.representations = reps
                    if mol.representations is not None:
                        for rep in mol.representations:
                            rep.parent = mol
            yield chunk

    def apply(
            self,
            func: callable,
            func_args: list | None = None,
            func_kwargs: dict | None = None,
            on_props: tuple[str, ...] | None = None,
            chunk_type: Literal["mol", "smiles", "rdkit", "df"] = "mol",
            chunk_processor: ParallelGenerator | None = None,
            no_parallel: bool = False,
    ) -> Generator[Iterable[Any], None, None]:
        """Apply a function to the molecules in the data frame.

        Args:
            func (callable): Function to apply to the molecules.
            func_args (list, optional): Additional arguments to pass to the function.
            func_kwargs (dict, optional):
                Additional keyword arguments to pass to the function.
            on_props (tuple, optional):
                Properties to pass to the function. If `None`, all properties will be
                passed.
            chunk_type (str, optional):
                Type of molecule to send to the function. Can be 'smiles', 'mol', or
                'rdkit'. Defaults to 'mol', which implies `TabularMol` objects.
            chunk_processor (ParallelGenerator, optional):
                The parallel generator to use for processing. If not specified,
                `self.chunkProcessor` is used.
            no_parallel (bool, optional):
                Whether to use parallel processing. Defaults to `False`.

        Returns:
            (Generator):
                A generator that yields the results of the supplied function on the
                chunked molecules from the data set.
        """
        chunk_processor = chunk_processor or self.chunkProcessor
        func_args = func_args or []
        func_kwargs = func_kwargs or {}
        if self.representations.nJobs > 1 and not no_parallel:
            for result in chunk_processor(
                    self.iterChunks(
                        self.chunkSize, chunk_type=chunk_type, on_props=on_props
                    ),
                    func,
                    *func_args,
                    **func_kwargs,
            ):
                yield result
        else:
            # do not use the parallel generator if n_jobs is 1
            for chunk in self.iterChunks(
                    self.chunkSize, chunk_type=chunk_type, on_props=on_props
            ):
                yield func(chunk, *func_args, **func_kwargs)

    @property
    def idProp(self) -> str:
        return self.storage.idProp

    def getProperty(self, name: str, ids: tuple[str] | None = None) -> Iterable[Any]:
        return self.representations.getProperty(name, ids)

    def getProperties(self) -> list[str]:
        return self.representations.getProperties()

    def addProperty(self, name: str, data: Sized, ids: list[str] | None = None):
        return self.representations.addProperty(name, data, ids)

    def hasProperty(self, name: str) -> bool:
        return self.representations.hasProperty(name)

    def removeProperty(self, name: str):
        return self.representations.removeProperty(name)

    def getSubset(self, subset: Iterable[str],
                  ids: Iterable[str] | None = None) -> PandasChemStore:
        return self.representations.getSubset(subset, ids)

    def getDF(self) -> pd.DataFrame:
        return self.representations.getDF()

    def dropEntries(self, ids: Iterable[str]):
        return self.representations.dropEntries(ids)

    def addEntries(self, ids: list[str], props: dict[str, list],
                   raise_on_existing: bool = True):
        assert "parent_id" in props, \
            "Parent ID is missing. It must be provided as 'parent_id' property."
        assert "sdf" in props, \
            "SDF table is missing. It must be provided as 'sdf' property."
        # FIXME: add checks for the parent_id and sdf properties
        return self.representations.addEntries(ids, props, raise_on_existing)

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["representations"] = os.path.relpath(self.representations.save(),
                                                    self.baseDir)
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.representations = PandasChemStore.fromFile(
            os.path.join(self.baseDir, state["representations"])
        )

    def save(self) -> str:
        return self.toFile(self.metaFile)

    def reload(self):
        self.__dict__.update(self.fromFile(self.metaFile).__dict__)

    def clear(self):
        """Clear the storage."""
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    @property
    def chunkSize(self) -> int:
        return self.representations.chunkSize

    def searchWithSMARTS(self, patterns: list[str]) -> "SMARTSSearchable":
        return self.representations.searchWithSMARTS(patterns)

    def searchOnProperty(self, prop_name: str, values: list[float | int | str],
                         exact=False) -> "PropSearchable":
        return self.representations.searchOnProperty(prop_name, values, exact)

    def processMols(
            self,
            processor: MolProcessor,
            proc_args: Iterable[Any] | None = None,
            proc_kwargs: dict[str, Any] | None = None,
            mol_type: Literal["smiles", "mol", "rdkit"] = "mol",
            add_props: Iterable[str] | None = None,
            chunk_processor: ParallelGenerator | None = None,
    ) -> Generator:
        """Apply a function to the molecules in the data frame.
        The SMILES  or an RDKit molecule will be supplied as the first
        positional argument to the function. Additional properties
        to provide from the data set can be specified with 'add_props', which will be
        a dictionary supplied as an additional positional argument to the function.

        IMPORTANT: For successful parallel processing with `multiprocessing`,
        the processor must be picklable.
        Also note that
        the returned generator may only produce results as soon as they are ready,
        which means that the chunks of data may
        not be in the same order as the original data frame. However, you can pass the
        value of `idProp` in `add_props` to identify the processed molecules or
        use `MolProcessorWithID` as the processor.

        Args:
            processor (MolProcessor):
                `MolProcessor` object to use for processing.
            proc_args (list, optional):
                Any additional positional arguments to pass to the processor.
            proc_kwargs (dict, optional):
                Any additional keyword arguments to pass to the processor.
            mol_type (str, optional):
                Type of molecule to send to the processor. Can be 'smiles', 'mol', or
                'rdkit'. Defaults to 'mol', which implies `TabularMol` objects.
            add_props (list, optional):
                List of data set properties to send to the processor. If `None`, all
                properties will be sent.
            chunk_processor (ParallelGenerator, optional):
                The parallel generator to use for processing. If not specified,
                `self.chunkProcessor` is used.

        Returns:
            Generator:
                A generator that yields the results of the supplied processor on
                the chunked molecules from the data set.
        """
        proc_args = proc_args or ()
        proc_kwargs = proc_kwargs or {}
        if add_props is None:
            add_props = self.getProperties()
        else:
            add_props = list(add_props)
        add_props = add_props + list(processor.requiredProps)
        chunk_processor = chunk_processor or self.chunkProcessor
        for prop in add_props:
            if prop not in self.getProperties():
                raise ValueError(
                    f"Cannot apply function '{processor}' to {self.name} because "
                    f"it requires the property '{prop}', which is not present in the "
                    "data set."
                )
        for result in self.apply(
                processor,
                func_args=proc_args,
                func_kwargs=proc_kwargs,
                on_props=add_props,
                chunk_type=mol_type,
                chunk_processor=chunk_processor,
                no_parallel=not processor.supportsParallel,
        ):
            yield result

    @property
    def identifier(self) -> ChemIdentifier:
        return self.representations.identifier

    def applyIdentifier(self, identifier: ChemIdentifier):
        return self.representations.applyIdentifier(identifier)

    @property
    def standardizer(self) -> ChemStandardizer:
        return self.representations.standardizer

    def applyStandardizer(self, standardizer: ChemStandardizer):
        return self.representations.applyStandardizer(standardizer)

    def getSummary(self) -> pd.DataFrame:
        """Show the number of representations for each parent molecule."""
        reps = self.representations.getDF()
        return reps.groupby("parent_id").size().to_frame("num_reps")
