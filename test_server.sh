#!/bin/bash

# Default values
SERVER_URL=${1:-"http://localhost:8000"}
QUERY=${2:-"What is the capital of France?"}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Testing server at:${NC} $SERVER_URL"
echo -e "${BLUE}Query:${NC} $QUERY"
echo -e "${BLUE}Sending request...${NC}"

# Send POST request to server
response=$(curl -s -X POST "$SERVER_URL/process" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"$QUERY\"}")

# Check if request was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Request successful!${NC}"
    
    # Extract and format the final answer if available
    final_answer=$(echo $response | grep -o '"final_answer":"[^"]*"' | cut -d':' -f2- | tr -d '"')
    
    if [ ! -z "$final_answer" ]; then
        echo -e "\n${GREEN}=== FINAL ANSWER ===${NC}"
        echo "$final_answer"
        echo -e "${GREEN}===================${NC}\n"
    else
        # Extract direct answer as fallback
        direct_answer=$(echo $response | grep -o '"direct_answer":"[^"]*"' | cut -d':' -f2- | tr -d '"')
        
        if [ ! -z "$direct_answer" ]; then
            echo -e "\n${GREEN}=== DIRECT ANSWER ===${NC}"
            echo "$direct_answer"
            echo -e "${GREEN}=====================${NC}\n"
        else
            # Just show raw response
            echo -e "\n${GREEN}=== FULL RESPONSE ===${NC}"
            echo "$response" | python3 -m json.tool
            echo -e "${GREEN}=====================${NC}\n"
        fi
    fi
else
    echo "Error sending request to server"
fi
