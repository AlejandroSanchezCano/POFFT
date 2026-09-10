"""
===============================================================================
Title:      Fetch sequences from UniProt
Outline:    Wrapper for the UniProt REST API to fetch sequences in FASTA format
            using the ID mapping service.
Docs:       https://www.uniprot.org/api-documentation/idmapping
Author:     Alejandro Sánchez Cano
Date:       02/09/2026
===============================================================================
"""

# Built-in modules
import re
import time
import math
import json
from pathlib import Path

# Third-party modules
import requests
from tqdm import tqdm

# Custom modules
from misc.logger import logger

class UniProtJob:

    # Class variables
    API = "https://rest.uniprot.org"

    def __init__(self, accessions: list[str]):
        self.accessions = accessions
        self.id = None

    def submit(self) -> None:
        '''
        Posts a request to the UniProt REST API using the idmapping endpoint.
        It gathers the job ID.
        '''
        # Send request to UniProt API
        response = requests.post(
            url=f"{self.API}/idmapping/run",
            data={
                "from": "UniProtKB_AC-ID",
                "to": "UniProtKB",
                "ids": ",".join(self.accessions),
            }
        )

        # Raise exception if request failed (status >= 400)
        if not response.ok:
            logger.error(
                f"Failed to submit job: error {response.status_code} - "
                f"{response.text}"
            )
            response.raise_for_status()

        # Store job ID
        self.id = response.json()["jobId"]

    def wait(
        self,
        poll_interval: int = 5,
        max_wait_time: int = 300
    ) -> None:
        '''
        Polls the UniProt REST API to check the status of the job until it is
        finished or failed. Raises an exception if the job fails or if it takes
        longer than the specified max_wait_time.
        
        Parameters
        ----------
        poll_interval : int
            Time in seconds between each poll request.
        max_wait_time : int
            Maximum time in seconds to wait for the job to finish before 
            raising a TimeoutError.
        '''

        # Starting time
        start = time.time()

        while True:
            # Kill stuck jobs
            current = time.time()
            elapsed = current - start
            if elapsed > max_wait_time:
                raise TimeoutError(f"Job {self.id} timed out after {max_wait_time} seconds")

            # Check job status
            url = f"{self.API}/idmapping/status/{self.id}"
            response = requests.get(url=url)
            #response.raise_for_status()
            status = response.json().get("jobStatus", 'FINISHED')
            hh_mm_ss = time.strftime("%H:%M:%S", time.localtime())
            logger.info(f"Job {self.id} status: {status} @ {hh_mm_ss}")

            # Handle job status
            if status == 'FINISHED':
                return
            elif status in ('FAILED', 'ERROR'):
                msg = f"Job {self.id} status {status}: {response.json()}"
                raise RuntimeError(msg)

            # Wait
            time.sleep(poll_interval)

    def download(
        self, 
        size: int = 40,
        out_dir: str | Path = "."
    ) -> str:
        '''
        Downloads the results of the job using pagination. Results are obtained
        in FASTA format and JSON format (for metadata and debugging). The JSON
        files are saved in the specified output directory.

        Parameters
        ----------
        size : int
            Number of entries to retrieve per request. The UniProt API has a
            maximum limit of 500 entries per request. If the job has more than
            this number of entries, the function will paginate through the 
            results using the "Link" header provided by the API. 
        out_dir : str
            Directory where the JSON files will be saved. Each entry will be
            saved as a separate JSON file named after its query accession 
            number.

        Returns
        -------
        str
            The FASTA formatted string containing the sequences.
        '''
        # Define initial URL
        url = f"{self.API}/idmapping/uniprotkb/results/{self.id}"
        params_fasta = {"format": "fasta", "size": size}
        params_json = {"format": "json", "size": size}
        if size > 500:
            raise ValueError("Pagination size limit exceeded (> 500)")

        # Progress bar
        pbar = tqdm(
            desc="Downloading sequences", 
            unit="page", 
            total=math.ceil(len(self.accessions) / size),
        )

        # Paginate through results
        fasta_results = []
        while url:
            # JSON request
            response = requests.get(url=url, params=params_json)
            response.raise_for_status()
            results = response.json()
            for result in results.get("results", []):
                accession = result['from']
                with open(f"{out_dir}/{accession}.json", "w") as f:
                    json.dump(result, f, indent=4)
            # FASTA request
            response = requests.get(url=url, params=params_fasta)
            response.raise_for_status()
            fasta_results.append(response.text)
            # Update progress bar
            pbar.update(1)
            # Check for pagination
            url = None
            params = None
            link_header = response.headers.get("Link", "")
            for link in link_header.split(","):
                match = re.search(r'<([^>]+)>;\s*rel="next"', link)
                if match:
                    url = match.group(1)
                    break
        
        return "\n".join(fasta_results)

if __name__ == "__main__":
    # Example usage
    accessions = [
        "O82732",       # Active
        "P62204",       # Demerged to 3 active entries
        "A0A2H5NPF5",   # Deleted
        "O00597"        # Deleted
    ]
    job = UniProtJob(accessions)
    job.submit()
    job.wait(poll_interval=1, max_wait_time=300)
    print(job.download())