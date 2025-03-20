# Use official Python image
FROM python:3.11.9

### Step 1: Set up a new user with proper permissions
RUN useradd -m -u 1000 user

# Switch to the new user
USER user

# Set up environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy the app files and set ownership to the new user
COPY --chown=user . $HOME/app

### Step 2: Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .


### Step 3: Fix permissions
USER root
RUN chmod -R 777 $HOME/app
# Copy entrypoint script and set execute permissions
# COPY entrypoint.sh /home/user/app/entrypoint.sh
# RUN chmod +x /home/user/app/entrypoint.sh
USER user  # Switch back to non-root user

# Expose Flask (7860) and Streamlit (8501) ports
EXPOSE 7860



# Run both Flask API & Streamlit App
#CMD ["./entrypoint.sh"]
#CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:5000 app:app & streamlit run streamlit_app.py --server.port=7860 --server.address=0.0.0.0"]
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]