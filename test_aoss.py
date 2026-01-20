"""             episode_match = re.search(r'episode[_\s](\d+)', episode_file, re.IGNORECASE)
            if episode_match:
                episode = int(episode_match.group(1))
            else:
                # Try numeric pattern
                num_match = re.search(r'(\d+)', episode_file)
                episode = int(num_match.group(1)) if num_match else 1






            # First try: look for eXX or EXX pattern (most common in anime filenames)
            episode_match = re.search(r'[sS]\d+[eE](\d+)', episode_file, re.IGNORECASE)
            if episode_match:
                episode = int(episode_match.group(1))
            else:
                # Alternative: last number in filename before .srt
                num_match = re.search(r'(\d+)(?=\.srt$)', episode_file)
                episode = int(num_match.group(1)) if num_match else 1



            # Look for the number right after 'e' or 'E'
            episode_match = re.search(r'[eE](\d+)', episode_file, re.IGNORECASE)
            if episode_match:
                episode = int(episode_match.group(1))
            else:
                num_match = re.search(r'(\d+)', episode_file)
                episode = int(num_match.group(1)) if num_match else 1




aws opensearchserverless batch-get-collection \
  --names linguasync-episodes \
  --region us-east-1 """