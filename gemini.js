// server.js
// Import necessary modules
const express = require('express');
const multer = require('multer');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require('@google/generative-ai');
require('dotenv').config(); // To load environment variables from a .env file

// --- Configuration ---
const PORT = process.env.PORT || 3000;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL_NAME = "gemini-pro-vision"; // Model capable of understanding images

// --- Initialize Express App ---
const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// --- Configure Multer for File Uploads ---
// Multer will store files in memory, making them easily accessible as buffers
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 }, // Limit file size to 10MB
    fileFilter: (req, file, cb) => {
        // Accept only image files
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Not an image! Please upload an image file.'), false);
        }
    }
});

// --- Initialize Gemini AI ---
if (!GEMINI_API_KEY) {
    console.error("Error: GEMINI_API_KEY is not set. Please set it in your .env file.");
    process.exit(1); // Exit if API key is not found
}
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: GEMINI_MODEL_NAME });

// --- Helper function to convert buffer to Gemini FilePart ---
function fileToGenerativePart(buffer, mimeType) {
    return {
        inlineData: {
            data: buffer.toString("base64"),
            mimeType
        },
    };
}

// --- API Endpoint for Image Upload and OCR ---
app.post('/upload-ocr', upload.single('imageFile'), async (req, res) => {
    // 'imageFile' is the name attribute of the file input field in your form
    if (!req.file) {
        return res.status(400).json({ error: 'No image file uploaded.' });
    }

    try {
        console.log('Received image:', req.file.originalname, 'MIME type:', req.file.mimetype, 'Size:', req.file.size);

        // The prompt for Gemini
        const prompt = "Extract all text from this image. If no text is visible, please indicate that no text was found. Focus solely on transcribing the text accurately.";

        // Prepare the image part for the Gemini API
        const imagePart = fileToGenerativePart(req.file.buffer, req.file.mimetype);

        // Generation configuration (optional, but good for safety)
        const generationConfig = {
            temperature: 0.2, // Lower temperature for more deterministic output (good for OCR)
            topK: 1,
            topP: 1,
            maxOutputTokens: 2048,
        };

        // Safety settings (adjust as needed)
        const safetySettings = [
            { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
            { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
            { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
            { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
        ];

        console.log('Sending request to Gemini API...');
        // Call the Gemini API
        const result = await model.generateContent({
            contents: [{ role: "user", parts: [imagePart, {text: prompt}] }],
            generationConfig,
            safetySettings
        });

        if (!result || !result.response || !result.response.candidates || !result.response.candidates[0].content || !result.response.candidates[0].content.parts) {
            console.error("Gemini API response structure is unexpected:", result);
            return res.status(500).json({ error: 'Failed to process image with Gemini API due to unexpected response structure.' });
        }

        const responseParts = result.response.candidates[0].content.parts;
        let extractedText = "";
        if (responseParts && responseParts.length > 0 && responseParts[0].text) {
            extractedText = responseParts[0].text;
        } else {
            // This case might occur if the model couldn't extract text or the response format is different.
            // You might want to inspect `result.response` more closely if this happens.
            extractedText = "No text found or unable to extract text.";
            console.warn("Gemini API did not return text in the expected part. Full response:", JSON.stringify(result.response, null, 2));
        }


        console.log('Gemini API response received.');
        res.status(200).json({
            message: 'Image processed successfully.',
            originalFilename: req.file.originalname,
            mimeType: req.file.mimetype,
            extractedText: extractedText.trim()
        });

    } catch (error) {
        console.error('Error processing image with Gemini API:', error);
        // Check for specific Gemini API errors if available, e.g., related to safety settings
        if (error.response && error.response.promptFeedback) {
            console.error('Gemini API Prompt Feedback:', error.response.promptFeedback);
            return res.status(500).json({
                error: 'Failed to process image with Gemini API.',
                details: error.message,
                promptFeedback: error.response.promptFeedback
            });
        }
        res.status(500).json({ error: 'Failed to process image.', details: error.message });
    }
});

// --- Basic Error Handler for Multer ---
app.use((err, req, res, next) => {
    if (err instanceof multer.MulterError) {
        // A Multer error occurred when uploading.
        return res.status(400).json({ error: `Multer error: ${err.message}` });
    } else if (err) {
        // An unknown error occurred.
        return res.status(400).json({ error: err.message });
    }
    next();
});


// --- Start the server ---
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
    if (!GEMINI_API_KEY) {
        console.warn("Warning: GEMINI_API_KEY is not set. The API calls will fail. Please create a .env file with your API key.");
    } else {
        console.log("Gemini API Key loaded successfully.");
    }
});
