""" This module just registers the NOMAD parser for SPRKKR. """
import re
from nomad.config.models.plugins import ParserEntryPoint

class SprkkrParserEntryPoint(ParserEntryPoint):
    """ NOMAD parser class for SPRKKR """

    def load(self):
        from .sprkkr_parser import SprkkrParser
        return SprkkrParser(**self.dict())

header="""          **************************************************************
          *                                                            *
          *        ****   *****   *****   *    *  *    *  *****        *
          *       *    *  *    *  *    *  *   *   *   *   *    *       *
          *       *       *    *  *    *  *  *    *  *    *    *       *
          *        ****   *****   *****   * *     * *     *****        *
          *            *  *       *  *    ** *    ** *    *  *         *
          *       *    *  *       *   *   *   *   *   *   *   *        *
          *        ****   *       *    *  *    *  *    *  *    *       *
          *                                                            *
          *------------------------------------------------------------*"""


sprkkr_parser = SprkkrParserEntryPoint(
    name = 'SprKkrParser',
    description = 'SPRKKR parser from ASE2SPRKKR',
    mainfile_name_re = r'.*\.out',
    mainfile_contents_re = re.escape(header)
)
