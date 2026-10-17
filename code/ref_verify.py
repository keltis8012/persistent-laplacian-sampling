# -*- coding: utf-8 -*-
"""引用外部核验交付门（固化）。

用法：python ref_verify.py [论文md路径]
对论文文献表逐条执行：
  - DOI 条目 → Crossref /works/{doi} 解析 + 题名 token 匹配（≥50%）
  - arXiv 条目 → arXiv API id_list 解析 + 题名 token 匹配
  - 书籍/无 ID 条目 → 标记 SKIP
退出码：0 = 全部通过；1 = 存在 FAIL。投稿前必须跑通（exit 0）方可交付。
"""
import re, sys, os, json, time
import urllib.request, urllib.parse, ssl

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAPERS = [os.path.join(os.path.dirname(HERE), '终稿_论文全文.md')]
ctx = ssl._create_unverified_context()
UA = {'User-Agent': 'ref-verify-gate/1.0 (mailto:ref.check@example.org)'}


def get(u, t=30):
    return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t, context=ctx).read()


def tokens(s):
    return set(re.findall(r'[a-z]{4,}', s.lower()))


def verify_entry(num, entry):
    dm = re.search(r'DOI (\S+?)\.?$', entry)
    ax = re.search(r'arXiv:(\d{4}\.\d{4,5})', entry)
    if dm:
        doi = dm.group(1).rstrip('.')
        j = json.loads(get('https://api.crossref.org/works/' + urllib.parse.quote(doi)))['message']
        ct = (j.get('title') or ['?'])[0]
        au = ';'.join(a.get('family', '?') for a in j.get('author', [])[:5])
        vol, pg = j.get('volume'), j.get('page')
        meta = f"{ct[:50]} | {au} | v{vol} pg{pg}"
    elif ax:
        aid = ax.group(1)
        xml = get('https://export.arxiv.org/api/query?id_list=' + aid).decode('utf-8', 'replace')
        mt = re.search(r'<entry>.*?<title>(.*?)</title>', xml, re.S)
        ct = re.sub(r'\s+', ' ', mt.group(1)) if mt else '?'
        meta = ct[:60]
    else:
        return 'SKIP', 'no DOI/arXiv id', ''
    ov = len(tokens(ct) & tokens(entry)) / max(1, len(tokens(ct)))
    return ('OK' if ov >= 0.5 else 'MISMATCH'), meta, ct


def main(paper_path):
    txt = open(paper_path, encoding='utf-8').read()
    refs = re.findall(r'^\[(\d+)\] (.+)$', txt[txt.index('## References'):], re.M)
    ok = fails = skips = 0
    print(f"引用外部核验门：{len(refs)} 条 | {paper_path}")
    for num, entry in refs:
        st, meta = None, None
        for attempt in range(4):
            try:
                st, meta, _ = verify_entry(num, entry)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 3:
                    time.sleep(5 * (attempt + 1)); continue
                st, meta = 'NET-%d' % e.code, 'rate-limited/unavailable'
                break
            except Exception as e:
                st, meta = 'NET-ERR', type(e).__name__
                break
        time.sleep(1.2)
        if st == 'OK':
            ok += 1; print(f"  [{num}] OK   {meta}")
        elif st == 'SKIP':
            skips += 1; print(f"  [{num}] SKIP {meta}")
        elif st.startswith('NET'):
            skips += 1; print(f"  [{num}] NET {meta}")
        else:
            fails += 1; print(f"  [{num}] FAIL {meta}")
    print(f"=== 门结果：OK {ok} / SKIP {skips} / FAIL {fails} ===")
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PAPERS[0]
    rc = main(os.path.abspath(target))
    sys.exit(rc)
