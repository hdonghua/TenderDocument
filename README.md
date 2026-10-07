# 标书生成智能体

输入项目概要，先生成大纲，再按小节分段并发生成标书。用来应对几万字标书一次放不进模型上下文的问题。

可以配置多条 DeepSeek（同一把密钥、不同模型，或不同账号），也可以配置 Cursor，以及任意 OpenAI 兼容接口。历史标书放进本地知识库后，生成每个小节时会检索相近片段，只借鉴写法和结构。

未知的资质、报价、人员和证书不会被编造，正文里会写成 `【待补充：…】`。

## 环境

- Python 3.10 及以上
- DeepSeek API Key，或 Cursor API Key

## 安装和运行

在项目目录打开终端：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

浏览器打开 Streamlit 给出的本地地址。在 `.env` 里填写密钥：

```text
DEEPSEEK_API_KEY=sk-...
CURSOR_API_KEY=cursor_...
```

只用 DeepSeek 时，`CURSOR_API_KEY` 可以留空。要用 Cursor 通道时再安装：

```powershell
pip install -r requirements-cursor.txt
```

Cursor 密钥在 [Cursor Dashboard → Integrations](https://cursor.com/dashboard/integrations) 创建。这个通道每写一节都会启动一次本地 Agent，速度比 DeepSeek 慢，章节多时把并发设为 1。

## 配置多个模型

打开应用里的「模型配置」。默认有三条：

| 名称 | 提供商 | 模型 | 建议用途 |
| --- | --- | --- | --- |
| DeepSeek Chat（正文） | DeepSeek | `deepseek-chat` | 写各节正文 |
| DeepSeek Reasoner（大纲） | DeepSeek | `deepseek-reasoner` | 写大纲 |
| Cursor Composer | Cursor | `composer-2.5` | 可选 |

两条 DeepSeek 默认共用 `DEEPSEEK_API_KEY`。要再加一个账号或别的模型时，点「新建配置」：

- 提供商选 DeepSeek，模型名填 `deepseek-chat` 或 `deepseek-reasoner`
- 密钥可以写在这一条上，保存在本机 `config/secrets.yaml`，不会写进 `config/models.yaml`
- 也可以不填密钥，改填环境变量名，启动时从环境或 `.env` 读取

OpenAI 兼容接口用来接自建网关：提供商选「OpenAI 兼容接口」，Base URL 按服务商文档填写（一般是以 `/v1` 结尾、且不要自己再加 `/chat/completions`）。

生成页可以分别选择「大纲模型」和「正文模型」。

点「测试连接」确认密钥可用。Reasoner 不使用温度参数；如果正文被截断，把该模型的 `max_tokens` 调大，再用「仅空白或失败章节」重写。

## 怎么生成长文

1. 填写项目名称、概要，以及招标要求或评分办法。
2. 生成大纲。大纲是 JSON，解析后显示成可编辑表格，可以改标题、要点、字数，也可以增删行。
3. 按大纲并发生成正文。每一节单独请求模型，请求里只有项目概要、这一节的要点，以及最多几段历史标书片段。
4. 有子节的章只写短导语，具体内容写在子节里，避免同一段话生成两遍。
5. 单节建议不超过 2000 字。目标总字数很高时，把内容拆成更多小节，而不是把一节写到模型输出上限。
6. 某一节失败或明显偏短时，生成范围选「仅空白或失败章节」再跑一次。偏短的节会自动重写一稿。

全文导出为 Markdown 和 Word，文件在 `output/`。

## 知识库如何搭建

知识库是本地检索，不把历史标书上传到向量云。检索用结巴分词加 TF-IDF，不需要单独的向量模型。

### 1. 准备文件

支持 `.md`、`.txt`、`.docx`、`.pdf`。旧版 `.doc` 请另存为 `.docx`。扫描版 PDF 提不出文字，需要先做文字识别，否则重建索引时会被跳过。

放进知识库之前先脱敏：

- 删掉报价、身份证号、手机号、银行账号
- 删掉未公开的客户名称和合同编号
- 证书编号不要留在正文里

仓库自带一份虚构示例：`knowledge/raw/示例-信息化项目技术方案.md`。正式使用时换成你自己的历史标书，示例可以删除。

### 2. 放入目录或在页面上传

两种方式任选：

- 把文件拷到 `knowledge/raw/`，子目录也可以
- 在应用的「知识库」页上传

上传只是把文件存进 `knowledge/raw/`，此时还不能被检索。

### 3. 重建索引

在「知识库」页点击「重建索引」。程序会：

1. 读取每个文件的正文
2. 按大约 700 字切成重叠片段
3. 用结巴分词做 TF-IDF，索引写到 `knowledge/index/index.joblib`

新增、删除或更换资料后都要再点一次重建。生成前如果勾了「参考知识库」但索引还不存在，程序会用当前资料自动建一次。

页面下方的「试检索」用来确认查得准不准。例如输入「办件流转 对象存储」，应能命中示例方案里的架构和材料相关段落。

### 4. 生成时怎么用

勾选「参考知识库中的历史标书」后：

- 写大纲时，用项目概要检索少量片段，只参考结构
- 写每一节时，用小节标题和要点再检索，把片段放进该节的提示词

模型被要求不得照搬历史标书里的项目名称、金额、人员和证书。某一节实际参考了哪些文件，写完后可在「章节状态」里看到。

没有历史标书时不要勾选，或让检索结果为空，生成只依据你填的概要。

## 目录

```text
app.py                 Streamlit 入口
tender_agent/          大纲、分段生成、知识库、模型调用、Word 导出
config/models.yaml     模型清单（不含密钥）
config/secrets.yaml    本机密钥（自动生成，不要提交）
knowledge/raw/         历史标书原文
knowledge/index/       重建索引后的本地索引
output/                导出的标书
```

## 检查

不调用外部模型的单元测试：

```powershell
python -m unittest tests.test_core
```
