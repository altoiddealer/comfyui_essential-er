from comfy_api.latest import io

class PassOrNone(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PassOrNone",
            display_name="Pass or None Alt ◯",
            description="Passes the input through, or outputs None when no input is provided.",
            category="utilities/primitive",
            search_aliases=[
                "null",
                "nothing",
                "empty",
                "blank",
            ],
            inputs=[
                io.AnyType.Input("anything", tooltip="Passes the input through, or outputs None when no input is provided.", optional=True),
            ],
            outputs=[
                io.AnyType.Output("output"),
                io.Boolean.Output("is_none"),
            ],
        )

    @classmethod
    def execute(cls, anything=None):
        return io.NodeOutput(
            anything,
            anything is None,
        )
