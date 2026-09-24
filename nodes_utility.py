from comfy_api.latest import io

class PassOrNone(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        matchtype_template = io.MatchType.Template("value")
        input_template = io.MatchType.Input(
            "anything",
            template=matchtype_template,
            optional=True,
        )
        autogrow_template = io.Autogrow.TemplateNames(
            input=input_template,
            names=["anything"] + [f"default{i}" for i in range(99)],
            min=0,
        )
        return io.Schema(
            node_id="PassOrNone",
            display_name="Pass or None Alt ◯",
            category="utilities/logic",
            description="Passes the first non-None value through, or outputs None when all inputs are None/not provided.",
            search_aliases=["fallback", "null", "nothing", "empty", "blank"],
            inputs=[
                io.Autogrow.Input(
                    "values",
                    optional=True,
                    template=autogrow_template,
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=matchtype_template,
                    id="output",
                ),
                io.Boolean.Output("is_none"),
            ],
        )

    @classmethod
    def execute(
        cls,
        values: io.Autogrow.Type,
    ) -> io.NodeOutput:
        for value in values.values():
            if value is not None:
                return io.NodeOutput(value, False)

        return io.NodeOutput(None, True)
