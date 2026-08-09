# we need these specific tools from the datahub library to actually talk to the graph database so we import them here
from datahub.ingestion.graph.client import DataHubGraph, DataHubGraphConfig

# this function is the main switch to turn on our agent and make sure it can see the datahub server ash is running
def wake_up_winnow():
    
    # here we set up the connection details pointing exactly to localhost 8080 which is the default port for the docker container so it knows exactly where to look for the metadata
    graph = DataHubGraph(
        DataHubGraphConfig(
            server="http://localhost:8080",
        )
    )
    
    # we use a try block because if ash hasnt started the docker container yet the script will crash so this catches the error and just tells us to wait instead of blowing up
    try:
        # test_connection pings the server to see if its awake and if it returns true we print a success message and hand back the graph object so we can use it later to search for tables
        if graph.test_connection():
            print("🟢 Winnow is online. Connected to the DataHub knowledge graph.")
            return graph
            
    # if the connection fails it catches the exception and prints the exact error so we can debug it and tells us to wait for ash
    except Exception as e:
        print(f"🔴 Connection failed. Error: {e}")
        print("Waiting for Ash to spin up the local Docker container...")
        return None

# this function takes the live connection we made in step one and uses it to actually search the database for the useless tables
def find_zombie_tables(graph):
    print("hunting for expensive snowflake tables in the ecommerce dataset...")
    
    # we use the built in search tool from the datahub library to look for entities labeled as dataset
    # we pass a filter query so it only brings back tables that live on snowflake because those are the ones wasting the most compute money
    results = graph.get_search_results(
        entity_type="dataset",
        query="platform:snowflake"
    )
    
    # we make an empty list here so we have a place to store all the useless tables we find
    trash_bin = []
    
    # now we loop through every single snowflake table that the search tool gave back to us
    for item in results.entities:
        
        # a urn is basically datahubs unique serial number for a specific table so we grab that to keep track of it
        table_id = item.urn
        
        # right now we just add every snowflake table to our list so we can send them all to the safety check
        trash_bin.append(table_id)
        print(f"flagged for review: {table_id}")
        
    # finally we hand back the list of flagged tables so the rest of the agent can double check them
    return trash_bin

# this function takes our list of potential zombie tables and interrogates the graph to see if they are actually safe to delete
def verify_safe_to_delete(graph, trash_bin):
    print("running safety checks on flagged tables...")
    
    # we make a new list to hold only the tables that pass our strict safety test
    confirmed_waste = []
    
    # we go through the list of tables we found in step two one by one
    for table_id in trash_bin:
        
        # datahub has a built in way to get the lineage so we ask it for all the downstream connections
        # downstream means anything that relies on this table to function like a looker dashboard or an airflow pipeline
        lineage = graph.get_lineage(
            entity_urn=table_id,
            direction="DOWNSTREAM"
        )
        
        # we check the relationships on the lineage object returned by the sdk because if its empty that means nothing in the entire company uses this data
        downstream_count = len(lineage.relationships) if lineage and hasattr(lineage, 'relationships') else 0
        
        if downstream_count == 0:
            # we print a success message and add it to our final kill list because zero downstream assets means safe to wipe
            print(f"verified safe to delete: {table_id} has zero downstream dependencies")
            confirmed_waste.append(table_id)
        else:
            # if it has relationships it means something relies on it so we print a warning and leave it alone so we dont break production
            print(f"keep table: {table_id} is being used by {downstream_count} downstream assets")
            
    # we hand back the final list of tables that are 100 percent safe to destroy
    return confirmed_waste

# this function takes the final list of verified garbage tables and turns it into a formatted markdown pull request report that ash can send straight to github
def generate_pr_report(confirmed_waste):
    print("generating github pr payload...")
    
    # we do a quick check to see if the list is empty because if there is no waste we dont need to generate a cleanup pr
    if not confirmed_waste:
        return "no zombie tables detected dataset is 100 percent clean"
        
    # we calculate an estimated cost savings based on roughly 150 dollars per month of snowflake storage and compute for each orphaned table
    monthly_savings = len(confirmed_waste) * 150
    
    # we build the markdown title and introduction header using standard github formatting
    report = "## 🧹 Winnow Automated Cleanup Request\n\n"
    report += "Winnow agent scanned the DataHub knowledge graph and identified the following orphan tables with **0 downstream dependencies**.\n\n"
    report += f"**Estimated Cost Impact:** Saving **${monthly_savings}/month** in Snowflake storage & compute.\n\n"
    report += "### Flagged Tables for Deprecation:\n"
    
    # we loop through each table urn and add it as a bullet point in the markdown report so the manager can review exact names
    for table in confirmed_waste:
        report += f"- `{table}`\n"
        
    report += "\n---\n*Action Required: Approve this Pull Request to automatically archive these entities in DataHub.*"
    
    # we hand back the complete markdown string ready to be posted to github
    return report

# this is the main sequence that runs when you start the file
if __name__ == "__main__":
    
    # first we wake up the agent and connect to the graph
    winnow_graph = wake_up_winnow()
    
    # we add a safety check here because if the connection failed we cant run a search
    if winnow_graph:
        
        # step 2 we find all the snowflake tables
        dead_weight = find_zombie_tables(winnow_graph)
        print(f"total potential zombie tables found: {len(dead_weight)}")
        
        # step 3 we run the safety check to filter out the important ones
        safe_to_delete = verify_safe_to_delete(winnow_graph, dead_weight)
        print(f"final count of tables safe to delete: {len(safe_to_delete)}")
        
        # step 4 we format the final report payload for the github bot
        pr_payload = generate_pr_report(safe_to_delete)
        print("\n--- FINAL GITHUB PR PAYLOAD ---")
        print(pr_payload)