/**
 * 监控视频人物活动检测 - 前端交互逻辑
 */
(() => {
    // ──────────── DOM ────────────
    const $ = (s) => document.querySelector(s);
    const dropZone = $("#drop-zone");
    const fileInput = $("#file-input");
    const uploadSection = $("#upload-section");
    const analysisSection = $("#analysis-section");
    const videoPlayer = $("#video-player");
    const progressOverlay = $("#progress-overlay");
    const progressMessage = $("#progress-message");
    const progressBar = $("#progress-bar");
    const progressPct = $("#progress-pct");
    const segmentCount = $("#segment-count");
    const timelineCanvas = $("#timeline-canvas");
    const timelineStart = $("#timeline-start");
    const timelineEnd = $("#timeline-end");
    const segmentsList = $("#segments-list");
    const timeline = $("#timeline");

    let currentTaskId = null;
    let analysisData = null;

    // ──────────── 工具函数 ────────────
    function formatTime(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
        return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    // ──────────── 拖拽上传 ────────────
    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    ["dragenter", "dragover"].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length) handleFile(files[0]);
    });

    // ──────────── 上传文件 ────────────
    async function handleFile(file) {
        const formData = new FormData();
        formData.append("file", file);

        // 切换到分析视图
        uploadSection.classList.add("hidden");
        analysisSection.classList.remove("hidden");
        progressOverlay.classList.remove("fade-out");
        progressMessage.textContent = "正在上传视频...";
        progressBar.style.width = "0%";
        progressPct.textContent = "0%";

        try {
            const res = await fetch("/api/upload", { method: "POST", body: formData });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "上传失败");
            }
            const data = await res.json();
            currentTaskId = data.task_id;

            // 设置视频源
            videoPlayer.src = `/api/video/${currentTaskId}`;

            // 开始分析
            progressMessage.textContent = "上传完成，开始分析...";
            await startAnalysis();
        } catch (e) {
            progressMessage.textContent = `❌ 错误: ${e.message}`;
        }
    }

    // ──────────── 开始分析 & SSE ────────────
    async function startAnalysis() {
        const res = await fetch(`/api/analyze/${currentTaskId}`, { method: "POST" });
        if (!res.ok) throw new Error("启动分析失败");

        // SSE 监听进度
        const evtSource = new EventSource(`/api/status/${currentTaskId}`);
        evtSource.onmessage = (e) => {
            const data = JSON.parse(e.data);
            progressBar.style.width = data.progress + "%";
            progressPct.textContent = Math.round(data.progress) + "%";
            progressMessage.textContent = data.message || "分析中...";

            if (data.status === "completed") {
                evtSource.close();
                onAnalysisComplete();
            } else if (data.status === "failed") {
                evtSource.close();
                progressMessage.textContent = `❌ 分析失败: ${data.message}`;
            }
        };
        evtSource.onerror = () => {
            evtSource.close();
        };
    }

    // ──────────── 分析完成 ────────────
    async function onAnalysisComplete() {
        // 淡出进度覆盖层
        progressOverlay.classList.add("fade-out");

        // 获取结果
        const res = await fetch(`/api/result/${currentTaskId}`);
        analysisData = await res.json();

        // 更新时间轴标签
        timelineEnd.textContent = formatTime(analysisData.video_duration);
        segmentCount.textContent = `${analysisData.total_segments} 段活动`;

        // 绘制时间轴
        drawTimeline();

        // 渲染活动列表
        renderSegments();

        // 窗口大小变化时重绘
        window.addEventListener("resize", drawTimeline);
    }

    // ──────────── 时间轴绘制 ────────────
    function drawTimeline() {
        if (!analysisData) return;

        const rect = timeline.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        timelineCanvas.width = rect.width * dpr;
        timelineCanvas.height = rect.height * dpr;

        const ctx = timelineCanvas.getContext("2d");
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        const duration = analysisData.video_duration;

        // 背景
        ctx.fillStyle = "#111827";
        ctx.fillRect(0, 0, w, h);

        // 网格线
        ctx.strokeStyle = "#1e293b";
        ctx.lineWidth = 1;
        const gridInterval = duration > 600 ? 60 : duration > 60 ? 10 : 5;
        for (let t = 0; t <= duration; t += gridInterval) {
            const x = (t / duration) * w;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }

        // 活动片段
        if (analysisData.segments) {
            analysisData.segments.forEach((seg) => {
                const x1 = (seg.start_time / duration) * w;
                const x2 = (seg.end_time / duration) * w;
                const segW = Math.max(x2 - x1, 3); // 最小3像素

                // 渐变填充
                const grad = ctx.createLinearGradient(x1, 0, x1, h);
                grad.addColorStop(0, "rgba(99, 102, 241, 0.8)");
                grad.addColorStop(1, "rgba(139, 92, 246, 0.6)");
                ctx.fillStyle = grad;

                // 圆角矩形
                const r = Math.min(4, segW / 2);
                ctx.beginPath();
                ctx.moveTo(x1 + r, 4);
                ctx.lineTo(x1 + segW - r, 4);
                ctx.quadraticCurveTo(x1 + segW, 4, x1 + segW, 4 + r);
                ctx.lineTo(x1 + segW, h - 4 - r);
                ctx.quadraticCurveTo(x1 + segW, h - 4, x1 + segW - r, h - 4);
                ctx.lineTo(x1 + r, h - 4);
                ctx.quadraticCurveTo(x1, h - 4, x1, h - 4 - r);
                ctx.lineTo(x1, 4 + r);
                ctx.quadraticCurveTo(x1, 4, x1 + r, 4);
                ctx.fill();

                // 发光
                ctx.shadowColor = "rgba(99, 102, 241, 0.4)";
                ctx.shadowBlur = 8;
                ctx.fill();
                ctx.shadowBlur = 0;
            });
        }

        // 当前播放位置指示线
        drawPlayhead(ctx, w, h, duration);
    }

    function drawPlayhead(ctx, w, h, duration) {
        const t = videoPlayer.currentTime || 0;
        const x = (t / duration) * w;
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();

        // 三角形指示
        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        ctx.moveTo(x - 5, 0);
        ctx.lineTo(x + 5, 0);
        ctx.lineTo(x, 6);
        ctx.fill();
    }

    // 视频播放时更新时间轴
    videoPlayer.addEventListener("timeupdate", () => {
        if (analysisData) drawTimeline();
    });

    // 点击时间轴跳转
    timeline.addEventListener("click", (e) => {
        if (!analysisData) return;
        const rect = timeline.getBoundingClientRect();
        const ratio = (e.clientX - rect.left) / rect.width;
        const targetTime = ratio * analysisData.video_duration;
        videoPlayer.currentTime = targetTime;
        videoPlayer.play();
    });

    // ──────────── 活动列表渲染 ────────────
    function renderSegments() {
        if (!analysisData || !analysisData.segments.length) {
            segmentsList.innerHTML = '<p class="segments-empty">未检测到人物活动</p>';
            return;
        }

        segmentsList.innerHTML = "";

        analysisData.segments.forEach((seg, i) => {
            const card = document.createElement("div");
            card.className = "segment-card";
            card.addEventListener("click", () => {
                videoPlayer.currentTime = seg.start_time;
                videoPlayer.play();
                // 滚动到播放器位置
                videoPlayer.scrollIntoView({ behavior: "smooth", block: "center" });
            });

            const thumbUrl = `/api/frame/${currentTaskId}/${seg.thumbnail_timestamp}`;

            card.innerHTML = `
                <div class="segment-index">${i + 1}</div>
                <div class="segment-thumb">
                    <img src="${thumbUrl}" alt="活动片段 ${i + 1}" loading="lazy">
                </div>
                <div class="segment-info">
                    <div class="segment-time">
                        <span class="segment-time-label">${formatTime(seg.start_time)} — ${formatTime(seg.end_time)}</span>
                        <span class="segment-duration">${seg.duration.toFixed(1)}s</span>
                        <span class="segment-persons">👤 ${seg.max_persons}人</span>
                    </div>
                    <p class="segment-desc">${seg.description}</p>
                </div>
            `;

            segmentsList.appendChild(card);
        });
    }
})();
