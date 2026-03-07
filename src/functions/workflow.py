from llama_index.core.workflow import Event
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent


# This event carries the data from the extraction step to the ranking step
class ExtractedDataEvent(Event):
    extracted_candidates: list
    job_criteria: str





class ResumeRankingWorkflow(Workflow):
    def __init__(self, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm # Pass your Qwen LLM in so the workflow can use it

    @step
    async def extract_resumes(self, ev: StartEvent) -> ExtractedDataEvent:
        print(f"📄 Extracting data from {len(ev.file_paths)} resumes...")
        
        # 1. Use your batch_extract_pdfs function here to get the raw text
        # 2. (Optional) Run a quick LLM extraction to get structured skills/experience
        
        # Mocking the extracted data for this example
        extracted_data = [{"file": path, "content": "Parsed text..."} for path in ev.file_paths]
        
        # Pass the extracted data to the next step
        return ExtractedDataEvent(
            extracted_candidates=extracted_data, 
            job_criteria=ev.criteria
        )

    @step
    async def rank_candidates(self, ev: ExtractedDataEvent) -> StopEvent:
        print(f"🏆 Ranking candidates for: {ev.job_criteria}")
        
        # 1. Send all the extracted data to the LLM and ask it to evaluate them
        prompt = (
            f"You are an expert recruiter. Rank the following candidates based strictly "
            f"on this job criteria: '{ev.job_criteria}'.\n\n"
            f"Candidate Data: {ev.extracted_candidates}\n\n"
            f"Provide a numbered leaderboard with a 1-sentence justification for each."
        )
        
        ranked_output = await self.llm.acomplete(prompt)
        
        # 2. StopEvent ends the workflow and returns the final text back to the agent
        return StopEvent(result=str(ranked_output))
    




import os
from llama_index.core.tools import FunctionTool

async def trigger_ranking_workflow(directory_path: str, criteria: str) -> str:
    """
    Triggers a heavy workflow to read resumes from a directory, extract their information, 
    and rank them based on the given job description or criteria.
    
    Use this tool ONLY when the user explicitly asks to rank, evaluate, or compare 
    a batch of resumes.
    
    Args:
        directory_path (str): The local folder path containing the resumes.
        criteria (str): The role or specific skills we are ranking them on.
    """
    print("🤖 Agent decided to trigger the Ranking Workflow!")
    
    # 1. Gather the files
    pdf_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.pdf')]
    
    # 2. Initialize and run the workflow
    # We set a high timeout because reading PDFs and ranking takes time!
    workflow = ResumeRankingWorkflow(llm=llm, timeout=300.0)
    final_ranking = await workflow.run(file_paths=pdf_paths, criteria=criteria)
    
    return final_ranking

# Turn it into a LlamaIndex Tool
ranking_tool = FunctionTool.from_defaults(async_fn=trigger_ranking_workflow)


