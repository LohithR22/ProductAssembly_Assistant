import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// No changes to ChatBubble
// Add the 'isHtml = false' prop
function ChatBubble({ content, who = 'bot', isHtml = false }) {
  if (who === 'bot') {
    return (
      <div className="bubble-row bot-row">
        <div className="avatar" aria-hidden>PA</div>
        <div className={`bubble ${who}`}>
          <div className="bubble-content">
            {/* ⭐ ADD THIS CHECK: */}
            {isHtml ? (
              <div dangerouslySetInnerHTML={{ __html: content || '' }} />
            ) : (
              <ReactMarkdown>{content || ''}</ReactMarkdown>
            )}
          </div>
        </div>
      </div>
    )
  }
  // user bubble unchanged
  return (
    <div className={`bubble ${who}`}>
      <div className="bubble-content" dangerouslySetInnerHTML={{ __html: content || '' }} />
    </div>
  )
}

export default function App() {
  const [file, setFile] = useState(null)
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState([])
  const [chatHistory, setChatHistory] = useState([])
  const fileRef = useRef()
  const chatRef = useRef()
  const [previewUrl, setPreviewUrl] = useState(null)
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light')

  // ⭐ ADD THIS: State for drag-and-drop overlay
  const [isDragging, setIsDragging] = useState(false)

  useEffect(() => {
    document.body.className = theme
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'))
  }

  const onSend = async (e) => {
    // Check for event (to prevent default)
    if (e) e.preventDefault()

    // Stop if loading or no content
    if (loading || (!file && !text)) return
    
    setLoading(true)

    // --- ⭐ START: Capture state and clear inputs immediately ---
    const userContent = text
    const userFile = file
    const currentPreviewUrl = previewUrl
    const isFileOnly = userFile && !userContent

    // Clear inputs now
    setText('')
    setFile(null)
    setPreviewUrl(null)
    if (fileRef.current) fileRef.current.value = null
    // --- ⭐ END: Clear inputs ---

    const nextChatHistory = [...chatHistory, { role: 'user', content: userContent || (isFileOnly ? '[image]' : '') }]
    setChatHistory(nextChatHistory)

    // Use the captured 'userContent'
    if (userContent) {
      const userMsg = `<pre style="white-space:pre-wrap">${escapeHtml(userContent)}</pre>`
      setMessages((m) => [...m, { who: 'user', content: userMsg }])
    }

    // Use captured 'userFile' and 'currentPreviewUrl'
    if (userFile && userContent && currentPreviewUrl) {
      const imgHtml = `<div class='user-image'><img src='${currentPreviewUrl}' alt='uploaded' /></div>`
      setMessages((m) => [...m, { who: 'user', content: imgHtml }])
    }

    try {
      const fd = new FormData()
      // Use captured 'userFile' and 'userContent'
      if (userFile) fd.append('image', userFile, userFile.name)
      if (userContent) fd.append('user_text', userContent)
      fd.append('chat_history', JSON.stringify(nextChatHistory))

      const res = await fetch(`${API_URL}/assist-multipart`, { method: 'POST', body: fd })
      if (!res.ok) {
        const txt = await res.text()
        const errHtml = `<div class='error'>Error ${res.status}: ${escapeHtml(txt)}</div>`
        setMessages((m) => [...m, { who: 'bot', content: errHtml, isHtml: true }]) // Use isHtml for errors too
        setLoading(false)
        return
      }

      const data = await res.json()
      const out = data?.generator_output

      // Use captured 'isFileOnly' and 'currentPreviewUrl'
      if (isFileOnly && currentPreviewUrl) {
        const imgHtml = `<div class='user-image'><img src='${currentPreviewUrl}' alt='uploaded' /></div>`
        setMessages((m) => [...m, { who: 'user', content: imgHtml }])
      }

      const botMarkdown = out?.raw_text || (typeof out === 'string' ? out : "Sorry, I had trouble generating a response.")
      setMessages((m) => [...m, { who: 'bot', content: botMarkdown }]) // This correctly uses Markdown
      setChatHistory((h) => [...nextChatHistory, { role: 'assistant', content: botMarkdown }])
      
      // The input clearing logic is no longer needed here
    } catch (err) {
      setMessages((m) => [...m, { who: 'bot', content: `<div class='error'>Request failed: ${escapeHtml(String(err))}</div>`, isHtml: true }]) // Use isHtml for errors
    } finally {
      setLoading(false)
    }
  }

  const onFileChange = (e) => {
    const f = e.target.files && e.target.files[0]
    handleFile(f)
  }

  // ⭐ ADD THIS: Central file handling logic
  const handleFile = (f) => {
    if (f && f.type.startsWith('image/')) {
      setFile(f)
      try {
        const url = URL.createObjectURL(f)
        setPreviewUrl(url)
      } catch (err) {
        setPreviewUrl(null)
      }
    } else {
      // Handle non-image file error if you want
      setFile(null)
      setPreviewUrl(null)
    }
  }

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight
    }
  }, [messages])

  // ⭐ ADD THIS: 'Enter' to send handler
  const handleKeyDown = (e) => {
    // If Enter is pressed (but not Shift+Enter for newline)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault() // Stop newline
      onSend(e) // Trigger send
    }
  }

  // --- ⭐ ADD THESE: Drag and Drop Handlers ---

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files && e.dataTransfer.files[0]
    handleFile(f) // Use our new central file handler
  }

  function escapeHtml(unsafe) {
    return unsafe.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  }

  return (
    // ⭐ UPDATE THIS: Add drag-and-drop handlers to the main app div
    <div
      className="app"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <header>
        <div>
          <h1>Product Assembly Assistant</h1>
          <p className="subtitle">Upload a photo and/or paste a short manual excerpt. The assistant will return the next immediate action as concise steps.</p>
        </div>
        <button className="theme-toggle" onClick={toggleTheme} title="Toggle theme">
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
      </header>

      <main>
        <div className="chat" ref={chatRef}>
          {messages.length === 0 && (
            <div className="welcome">
              <h3>Welcome</h3>
              <p>Upload a photo or paste an excerpt and press <strong>Send</strong>. You can also drag and drop an image.</p>
            </div>
          )}

          {messages.map((m, i) => (
            <ChatBubble key={i} who={m.who} content={m.content} />
          ))}

          {loading && (
            // ⭐ ADD: isHtml={true}
            <ChatBubble who="bot" content={`<div class='typing'><span></span><span></span><span></span></div>`} isHtml={true} />
          )}
        </div>

        <form className="composer" onSubmit={onSend}>
          <div className="inputs">
            <div className="file-col">
              <input ref={fileRef} id="image" type="file" accept="image/*" onChange={onFileChange} style={{ display: 'none' }} />
              {previewUrl && (
                <div className="preview">
                  <img src={previewUrl} alt="preview" />
                </div>
              )}
            </div>
            <div className="prompt-row">
              {/* ⭐ UPDATE THIS: Add the onKeyDown handler */}
              <textarea
                placeholder="Paste manual excerpt or ask a short question"
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              {/* ⭐ UPDATE THIS: Replace emoji with a 'nice' SVG icon */}
              <button type="button" className="attach" onClick={() => fileRef.current && fileRef.current.click()} title="Attach image">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  width="20px"
                  height="20px"
                >
                  <path
                    fillRule="evenodd"
                    d="M18.97 3.659a2.25 2.25 0 00-3.182 0l-10.12 10.12a2.25 2.25 0 000 3.182c.878.878 2.304.878 3.182 0l10.12-10.12a.75.75 0 111.06 1.06l-10.12 10.12a3.75 3.75 0 01-5.304-5.304l10.12-10.12a2.25 2.25 0 003.182 0l.707.707a.75.75 0 01-1.06 1.06l-.707-.707zM6.12 17.88a.75.75 0 01-1.06-1.06l10.12-10.12a.75.75 0 011.06 1.06L6.12 17.88z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            </div>
          </div>

          <div className="actions">
            <button type="submit" className="send" disabled={loading}>{loading ? 'Thinking…' : 'Send'}</button>
          </div>
        </form>
      </main>


      {/* ⭐ ADD THIS: The drag-and-drop overlay */}
      <div className={`drop-zone-overlay ${isDragging ? 'visible' : ''}`}>
        <div className="drop-zone-content">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
          </svg>
          <p>Drop your image here</p>
        </div>
      </div>

    </div>
  )
}
