# Function to read the txt file and parse the content into a dictionary
def parse_trajectory_file(file_path):
    trajectory_dict = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Parse lines
        flag_into_dict = False
        for line in lines:
            if line.strip() == "FINISHED":
                break  # Stop parsing at the "FINISHED" marker
            if line.strip() == "```python":
                flag_into_dict = True
            if ":" in line and flag_into_dict:
                key, content = line.strip().split(":", 1)
                key = int(key)
                trajectory_dict[key] = content.strip().strip('",')
                
    return trajectory_dict

# Function to replace "same" with the last valid content in the dictionary
def replace_same_content(data):
    last_valid_content = None
    for key, value in data.items():
        if value != "same":
            last_valid_content = value
        else:
            data[key] = last_valid_content
    return data

# Function to save the updated dictionary back to a txt file
def save_trajectory_file(data, output_file_path):
    with open(output_file_path, 'w') as file:
        for key, value in data.items():
            file.write(f"{key}: {value}\n")
        file.write("FINISHED\n")


# # Example of usage:
# # Load the data from the input file
# input_file_path = '/home/lixiang/codebase/embodied-CoT/scripts/generate_embodied_data/prompt_answer.txt'  # replace with your file path
# trajectory_data = parse_trajectory_file(input_file_path)

# # Replace "same" entries
# updated_trajectory_data = replace_same_content(trajectory_data)

# # Save the updated data to a new file
# output_file_path = '/home/lixiang/codebase/embodied-CoT/scripts/generate_embodied_data/updated_trajectory_reasoning.txt'  # replace with your desired output file path
# save_trajectory_file(updated_trajectory_data, output_file_path)