from pathlib import Path
import yaml

class UserOverrides:

    def __init__(
        self,
        filename="user_node-defaults.yaml",
        directory=None,
    ):
        if directory is None:
            directory = Path(__file__).resolve().parent

        self.path = Path(directory) / filename
        self.data = self._load()

    def _load(self):

        if not self.path.is_file():
            return {}

        try:
            with self.path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = yaml.safe_load(file)

        except Exception as error:

            print(
                "[UserOverrides] "
                f"Failed to load {self.path.name}: {error}"
            )

            return {}

        if not isinstance(data, dict):

            print(
                "[UserOverrides] "
                f"{self.path.name} must contain a YAML mapping."
            )

            return {}

        return data

    def get(
        self,
        *path,
        default=None,
    ):
        value = self.data

        for key in path:

            if not isinstance(value, dict):
                return default

            if key not in value:
                return default

            value = value[key]

        if type(value) is not type(default):
            return default

        return value
