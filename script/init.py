from dataclasses import dataclass
import os
import importlib.resources
from pathlib import Path
from typing import ClassVar
from jinja2 import Environment, FileSystemLoader, Template

import distflow_schema
from general_function import snake_to_camel

dirname = Path(__file__).parent
env = Environment(loader=FileSystemLoader(dirname / "templates/"))


@dataclass
class DataModel:
    name: str
    filename: str

@dataclass
class InputSchemaInit:
    package_template: ClassVar[Template] = env.get_template("input_model_package.py.j2")
    # schema_template: ClassVar[Template] = env.get_template("raw_data_schema.py.j2")

    def render_package(self, models_list: list[DataModel]) -> str:
            return self.package_template.render(models_list=models_list)

    # def render_schema(self, models_list: list[RawDataModel]) -> str:
    #         return self.schema_template.render(models_list=models_list)

    def save(self):
                
        model_filename= importlib.resources.files(distflow_schema).joinpath("__init__.py")
        model_foldername = importlib.resources.files(distflow_schema)
        print(model_foldername)
        # schema_filename = importlib.resources.files(topology_models).joinpath("schema.py")
        models_list: list[DataModel] = []
        
        for file in os.listdir(model_foldername): # type: ignore
            if not file.startswith("_"):
                file = file.replace(".py", "")
                models_list.append(DataModel(name=snake_to_camel(file), filename=file))
    
        with open(model_filename, mode="w", encoding="utf-8") as message: # type: ignore
            message.write(self.render_package(models_list=models_list))
            print(f"... wrote {model_filename}")
        
        # with open(schema_filename, mode="w", encoding="utf-8") as message: # type: ignore
        #     message.write(self.render_schema(models_list=models_list))
        #     print(f"... wrote {schema_filename}")


if __name__ == "__main__":
    print("Hello world")
    InputSchemaInit().save()
