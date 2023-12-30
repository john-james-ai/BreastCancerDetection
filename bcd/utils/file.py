#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /bcd/utils/file.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday April 4th 2023 08:46:04 pm                                                  #
# Modified   : Saturday December 30th 2023 09:51:12 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import codecs
import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, List, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


# ------------------------------------------------------------------------------------------------ #
def getsize(filepath: str) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = os.stat(filepath).st_size
    idx = 0
    while size > 1000:
        idx += 1
        size /= 1024
    return f"{round(size,2)} {units[idx]}"


# ------------------------------------------------------------------------------------------------ #


class IO(ABC):  # pragma: no cover
    """Base class for IO Files"""

    _logger = logging.getLogger(
        f"{__module__}.{__name__}",
    )

    @classmethod
    def read(cls, filepath: str, **kwargs) -> Any:
        data = cls._read(filepath, **kwargs)
        return data

    @classmethod
    @abstractmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        pass

    @classmethod
    def write(cls, filepath: str, data: Any, **kwargs) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cls._write(filepath, data, **kwargs)

    @classmethod
    @abstractmethod
    def _write(cls, filepath: str, data: Any, **kwargs) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #
#                                         EXCEL IO                                                 #
# ------------------------------------------------------------------------------------------------ #


class ExcelIO(IO):  # pragma: no cover
    """Reads and writes Excel Files"""

    @classmethod
    def _read(
        cls,
        filepath: str,
        sheet_name: Union[str, int, list, None] = 0,
        header: Union[int, None] = 0,
        index_col: Union[int, str] = None,
        usecols: List[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        return pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            usecols=usecols,
            **kwargs,
        )

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: pd.DataFrame,
        sheet_name: str = "Sheet1",
        columns: Union[str, list] = None,
        header: Union[bool, list] = True,
        index: bool = False,
        **kwargs,
    ) -> None:
        data.to_excel(
            excel_writer=filepath,
            sheet_name=sheet_name,
            columns=columns,
            header=header,
            index=index,
            **kwargs,
        )


# ------------------------------------------------------------------------------------------------ #
#                                        CSV IO                                                    #
# ------------------------------------------------------------------------------------------------ #


class CSVIO(IO):  # pragma: no cover
    """Reads and writes CSV Files"""

    @classmethod
    def _read(
        cls,
        filepath: str,
        sep: str = ",",
        header: Union[int, None] = 0,
        index_col: Union[int, str] = None,
        usecols: List[str] = None,
        low_memory: bool = False,
        encoding: str = "utf-8",
        **kwargs,
    ) -> pd.DataFrame:
        return pd.read_csv(
            filepath,
            sep=sep,
            header=header,
            index_col=index_col,
            usecols=usecols,
            low_memory=low_memory,
            encoding=encoding,
        )

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: pd.DataFrame,
        sep: str = ",",
        index: bool = False,
        index_label: bool = None,
        encoding: str = "utf-8",
        **kwargs,
    ) -> None:
        data.to_csv(
            filepath,
            sep=sep,
            index=index,
            index_label=index_label,
            encoding=encoding,
            escapechar="\\",
        )


# ------------------------------------------------------------------------------------------------ #
#                                        TSV IO                                                    #
# ------------------------------------------------------------------------------------------------ #


class TSVIO(IO):  # pragma: no cover
    """Reads and writes TSV files."""

    @classmethod
    def _read(
        cls,
        filepath: str,
        sep: str = "\t",
        header: Union[int, None] = 0,
        index_col: Union[int, str] = None,
        usecols: List[str] = None,
        low_memory: bool = False,
        encoding: str = "utf-8",
        **kwargs,
    ) -> pd.DataFrame:
        return pd.read_csv(
            filepath,
            sep=sep,
            header=header,
            index_col=index_col,
            usecols=usecols,
            low_memory=low_memory,
            encoding=encoding,
        )

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: pd.DataFrame,
        sep: str = "\t",
        index: bool = False,
        index_label: bool = None,
        encoding: str = "utf-8",
        **kwargs,
    ) -> None:
        data.to_csv(
            filepath,
            sep=sep,
            index=index,
            index_label=index_label,
            encoding=encoding,
            escapechar="\\",
        )


# ------------------------------------------------------------------------------------------------ #
#                                        YAML IO                                                   #
# ------------------------------------------------------------------------------------------------ #


class YamlIO(IO):  # pragma: no cover
    """Reads and writes YAML files."""

    @classmethod
    def _read(cls, filepath: str, **kwargs) -> dict:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:  # pragma: no cover
                cls._logger.exception(e)
                raise IOError() from e
            finally:
                f.close()

    @classmethod
    def _write(cls, filepath: str, data: Any, **kwargs) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            try:
                yaml.dump(data, f)
            except yaml.YAMLError as e:  # pragma: no cover
                cls._logger.exception(e)
                raise IOError() from e
            finally:
                f.close()


# ------------------------------------------------------------------------------------------------ #
#                                         PICKLE                                                   #
# ------------------------------------------------------------------------------------------------ #


class PickleIO(IO):  # pragma: no cover
    """Reads and writes Pickle data."""

    @classmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        with open(filepath, "rb") as f:
            try:
                return pickle.load(f)
            except pickle.PickleError() as e:  # pragma: no cover
                cls._logger.exception(e)
                raise IOError() from e
            finally:
                f.close()

    @classmethod
    def _write(cls, filepath: str, data: Any, write_mode: str = "wb", **kwargs) -> None:
        # Note, "a+" write_mode for append. If <TypeError: write() argument must be str, not bytes>
        # use "ab+"
        with open(filepath, write_mode, encoding="utf-8") as f:
            try:
                pickle.dump(data, f)
            except pickle.PickleError() as e:  # pragma: no cover
                cls._logger.exception(e)
                raise
            finally:
                f.close()


# ------------------------------------------------------------------------------------------------ #
#                                         PARQUET                                                  #
# ------------------------------------------------------------------------------------------------ #


class ParquetIO(IO):  # pragma: no cover
    """Reads and writes Parquet files."""

    @classmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        """Read the pyarrow table, then convert to pandas."""
        table = pa.parquet.read_table(filepath, memory_map=True)
        return table.to_pandas()

    @classmethod
    def _write(cls, filepath: str, data: pd.DataFrame, **kwargs) -> None:
        """Converts Pandas DataFrame to a pyarrow table, then persists."""
        table = pa.Table.from_pandas(data)
        pq.write_table(table, filepath)


# ------------------------------------------------------------------------------------------------ #
#                                           HTML                                                   #
# ------------------------------------------------------------------------------------------------ #


class HtmlIO(IO):  # pragma: no cover
    """Reads and writes HTML"""

    @classmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        """Read the raw html."""
        file = codecs.open(filename=filepath, encoding="utf-8")
        return file.read()

    @classmethod
    def _write(cls, filepath: str, data: pd.DataFrame, **kwargs) -> None:
        """Converts Pandas DataFrame to a pyarrow table, then persists."""
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------ #
#                                          JSON                                                    #
# ------------------------------------------------------------------------------------------------ #


class JsonIO(IO):  # pragma: no cover
    """Reads and writes JSON files."""

    @classmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        """Read the parsed dictionary from a json file."""
        with open(filepath, "r", encoding="utf-8") as json_file:
            return json.load(json_file)

    @classmethod
    def _write(cls, filepath: str, data: dict, **kwargs) -> None:
        """Writes a dictionary to a json file."""
        with open(filepath, "w", encoding="utf-8") as json_file:
            if isinstance(data, list):
                for datum in data:
                    if isinstance(datum, dict):
                        json.dump(datum, json_file, indent=2)
                    else:
                        msg = "JsonIO supports dictionaries and lists of dictionaries only."
                        cls._logger.exception(msg)
                        raise ValueError(msg)
            else:
                try:
                    json.dump(data, json_file, indent=2)
                except json.JSONDecodeError as e:
                    cls._logger.exception(f"Exception of type {type(e)} occurred.\n{e}")
                    raise


# ------------------------------------------------------------------------------------------------ #
#                                       IO SERVICE                                                 #
# ------------------------------------------------------------------------------------------------ #
class IOService:  # pragma: no cover
    """File I/O Service"""

    __io = {
        "html": HtmlIO,
        "dat": CSVIO,
        "csv": CSVIO,
        "tsv": TSVIO,
        "yaml": YamlIO,
        "yml": YamlIO,
        "json": JsonIO,
        "pkl": PickleIO,
        "pickle": PickleIO,
        "xlsx": ExcelIO,
        "xls": ExcelIO,
        "parquet": ParquetIO,
    }
    _logger = logging.getLogger(
        f"{__module__}.{__name__}",
    )

    @classmethod
    def read(cls, filepath: str, **kwargs) -> Any:
        io = cls._get_io(filepath)
        return io.read(filepath, **kwargs)

    @classmethod
    def write(cls, filepath: str, data: Any, **kwargs) -> None:
        io = cls._get_io(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        io.write(filepath=filepath, data=data, **kwargs)

    @classmethod
    def _get_io(cls, filepath: str) -> IO:
        try:
            file_format = os.path.splitext(filepath)[1].replace(".", "")
            return IOService.__io[file_format]
        except TypeError as exc:
            if filepath is None:
                msg = "Filepath is None"
                cls._logger.exception(msg)
                raise ValueError(msg) from exc
        except KeyError as exc:
            msg = "File type {} is not supported.".format(file_format)
            cls._logger.exception(msg)
            raise ValueError(msg) from exc
