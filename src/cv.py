import asyncio
import random

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Event, Context
# from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI



llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")


class ProcessingEvent(Event):
    intermediate_result: str

class LoopEvent(Event):
    loop_output: str




class CVWorkflow(Workflow):
    @step()
    async def start(self, ctx: Context, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
       if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
       else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step()
    async def step_2(self, ctx: Context, ev: ProcessingEvent) -> StopEvent:
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)




# 2. Execution logic
async def main():
    cv_workflow = CVWorkflow(timeout=10, verbose=True) # verbose=True helps debugging
    # We pass the input data inside the run() method
    result = await cv_workflow.run()
    print(f"Final Result: {result}")

    # draw_all_possible_workflows(cv_workflow, "flow.html")


if __name__ == "__main__":
    asyncio.run(main())



