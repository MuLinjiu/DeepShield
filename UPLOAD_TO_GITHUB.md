# 如何上传到GitHub

## 📋 准备清单

✅ 已完成：
- [x] 创建 `.gitignore` 文件
- [x] 清理临时文件
- [x] 整理项目结构
- [x] 创建 LICENSE 文件
- [x] 创建专业的 README.md

## 🚀 上传步骤

### 1. 初始化Git仓库

```bash
cd /home/ubuntu/Workspace/DeepShield

# 初始化git
git init

# 添加所有文件
git add .

# 查看将要提交的文件
git status
```

### 2. 创建首次提交

```bash
# 提交所有文件
git commit -m "Initial commit: DeepShield LoRA Network Flow Security Classifier"
```

### 3. 在GitHub上创建仓库

1. 访问 https://github.com
2. 点击右上角 "+" -> "New repository"
3. 填写信息：
   - Repository name: `DeepShield`
   - Description: `Network Flow Security Classifier with LoRA Fine-tuning`
   - 选择 Public 或 Private
   - **不要**勾选 "Add a README file"（我们已经有了）
   - **不要**勾选 "Add .gitignore"（我们已经有了）
   - **不要**勾选 "Choose a license"（我们已经有了）
4. 点击 "Create repository"

### 4. 连接远程仓库并推送

```bash
# 添加远程仓库（替换成你的用户名）
git remote add origin https://github.com/YOUR_USERNAME/DeepShield.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

### 5. 如果需要认证

如果推送时要求输入用户名和密码：

```bash
# 配置git用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 使用Personal Access Token而不是密码
# 在GitHub Settings -> Developer settings -> Personal access tokens 创建token
# 然后使用token作为密码
```

## 📝 后续更新

当您修改了代码后：

```bash
# 查看修改
git status

# 添加修改的文件
git add .

# 提交
git commit -m "描述你的修改"

# 推送到GitHub
git push
```

## 🔒 保护数据安全

`.gitignore` 已配置忽略：
- ✅ 训练数据文件（`*.jsonl` in data/processed/）
- ✅ 模型checkpoint（`lora-*/`）
- ✅ 训练日志（`*.log`）
- ✅ 临时文件

**请确认不要上传：**
- 真实的网络流量数据
- 训练好的模型文件（太大）
- 包含敏感信息的配置文件

## ✅ 上传前最后检查

```bash
# 查看将要上传的文件
git ls-files

# 应该看到：
# - README.md
# - LICENSE
# - .gitignore
# - 训练脚本（.py）
# - 配置脚本（.sh）
# - requirements.txt
# - 示例数据（sample_*.jsonl）

# 不应该看到：
# - data/processed/*.jsonl（大数据文件）
# - lora-*（模型checkpoint）
# - *.log（日志文件）
```

## 🎯 推荐的GitHub仓库设置

上传后，在GitHub仓库页面：

1. **About**: 添加描述和标签
   - Topics: `deep-learning`, `network-security`, `lora`, `pytorch`, `intrusion-detection`

2. **README**: 确保README.md显示正常

3. **Releases**: 创建第一个版本
   - Tag: `v1.0.0`
   - Title: `Initial Release`

## 📢 分享您的项目

完成后，您的仓库链接将是：
```
https://github.com/YOUR_USERNAME/DeepShield
```

可以分享给：
- 研究社区
- 同事和collaborators
- 在论文中引用

---

## 🐛 常见问题

### Q: push时出现403错误
**A**: 使用Personal Access Token而不是密码

### Q: 文件太大无法上传
**A**: 检查 `.gitignore` 是否正确配置，使用 `git rm --cached` 移除已追踪的大文件

### Q: 想要忽略某个文件但它已经被追踪
**A**: 
```bash
git rm --cached filename
git commit -m "Remove tracked file"
```

---

**准备好了吗？开始上传到GitHub吧！** 🚀

