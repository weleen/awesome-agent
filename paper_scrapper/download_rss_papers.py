import os
import requests
from tqdm import tqdm
import re

def download_paper(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as file, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def extract_papers_from_md(md_content):
    # 使用正则表达式匹配标题和URL
    pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    matches = re.findall(pattern, md_content)
    return matches

def main():
    # 创建保存目录
    save_dir = "RSS2025"
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取Markdown文件内容
    with open('RSS2025_Papers.md', 'r') as f:
        content = f.read()
    
    # 提取所有论文信息
    papers = extract_papers_from_md(content)
    
    # 下载所有论文
    success_count = 0
    failed_papers = []
    
    for title, url in tqdm(papers, desc="Downloading papers"):
        # 从URL中提取文件名
        filename = url.split('/')[-1]
        save_path = os.path.join(save_dir, filename)
        
        if download_paper(url, save_path):
            success_count += 1
        else:
            failed_papers.append((title, url))
    
    # 打印下载统计
    print(f"\nDownload completed!")
    print(f"Successfully downloaded: {success_count}/{len(papers)} papers")
    
    if failed_papers:
        print("\nFailed to download the following papers:")
        for title, url in failed_papers:
            print(f"- {title}\n  {url}")

if __name__ == "__main__":
    main()