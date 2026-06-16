"""NOMAD parser entry point for SPRKKR output files."""

from typing import Dict

from ase2sprkkr.outputs.task_result import TaskResult
from nomad.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser

from .tasks import SprkkrTaskParser  # also registers scf, dos parsers


class SprkkrParser(MatchingParser):

    def parse(
        self,
        mainfile: str,
        archive: EntryArchive,
        logger=None,
        child_archives: Dict[str, EntryArchive] = None,
    ) -> None:
        output = TaskResult.from_file(mainfile)
        parser = SprkkrTaskParser.for_output(output)
        if parser is None:
            if logger:
                task = getattr(output, 'task_name', None)
                logger.warning(
                    f'No NOMAD parser registered for SPRKKR task '
                    f'{task!r} ({mainfile})'
                )
            return
        archive.data = parser.simulation()
