# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable
ENV PATH="/usr/src/app:${PATH}"

# Run the command when the container launches
CMD ["python", "./gcn.py"]
