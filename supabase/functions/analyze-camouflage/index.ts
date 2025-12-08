import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { image } = await req.json();
    
    if (!image) {
      throw new Error('No image provided');
    }

    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');
    if (!LOVABLE_API_KEY) {
      throw new Error('LOVABLE_API_KEY not configured');
    }

    console.log('Calling Lovable AI for camouflage analysis...');

    const response = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${LOVABLE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'google/gemini-2.5-pro',
        messages: [
          {
            role: 'system',
            content: `You are an expert wildlife biologist specializing in animal camouflage and adaptive coloration. 
            Analyze images to detect camouflaged animals, identify species, and quantify camouflage effectiveness.
            
            Provide detailed analysis including:
            - Whether camouflage is present (true/false)
            - Species identification if detected
            - Camouflage percentage (0-100)
            - AI confidence level (0-100)
            - Detailed description of findings
            - List of specific camouflage adaptations observed
            - Bounding box coordinates of the camouflaged animal (approximate percentage positions from 0-100)
            - List of specific camouflaged regions with their coordinates
            
            Respond ONLY with valid JSON in this exact format:
            {
              "detected": boolean,
              "species": "string or null",
              "camouflagePercentage": number,
              "confidence": number,
              "description": "string",
              "adaptations": ["string", "string"],
              "boundingBox": {
                "x": number (0-100, left position as percentage),
                "y": number (0-100, top position as percentage),
                "width": number (0-100, width as percentage),
                "height": number (0-100, height as percentage)
              },
              "camouflageRegions": [
                {
                  "description": "string (e.g., 'head and neck region')",
                  "intensity": number (0-100, how well camouflaged),
                  "x": number (0-100),
                  "y": number (0-100),
                  "width": number (0-100),
                  "height": number (0-100)
                }
              ]
            }`
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Analyze this image for camouflaged animals. Detect any animals using camouflage, identify the species if possible, and provide a detailed analysis of their camouflage techniques.'
              },
              {
                type: 'image_url',
                image_url: {
                  url: image
                }
              }
            ]
          }
        ],
        temperature: 0.3,
        max_tokens: 2000,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Lovable AI error:', errorText);
      throw new Error(`AI analysis failed: ${response.status}`);
    }

    const aiResponse = await response.json();
    console.log('AI response received');

    const content = aiResponse.choices?.[0]?.message?.content;
    if (!content) {
      throw new Error('No content in AI response');
    }

    // Parse the JSON response from the AI
    let analysisData;
    try {
      // Extract JSON from markdown code blocks if present
      const jsonMatch = content.match(/```json\n?([\s\S]*?)\n?```/) || content.match(/```\n?([\s\S]*?)\n?```/);
      const jsonString = jsonMatch ? jsonMatch[1] : content;
      analysisData = JSON.parse(jsonString);
    } catch (parseError) {
      console.error('Failed to parse AI response:', content);
      // Fallback response if parsing fails
      analysisData = {
        detected: false,
        species: null,
        camouflagePercentage: 0,
        confidence: 50,
        description: 'Unable to parse AI analysis. Please try again with a different image.',
        adaptations: [],
        boundingBox: null,
        camouflageRegions: []
      };
    }

    console.log('Analysis complete:', analysisData);

    return new Response(
      JSON.stringify(analysisData),
      { 
        headers: { 
          ...corsHeaders, 
          'Content-Type': 'application/json' 
        } 
      }
    );

  } catch (error) {
    console.error('Error in analyze-camouflage function:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return new Response(
      JSON.stringify({ 
        error: errorMessage,
        detected: false,
        species: null,
        camouflagePercentage: 0,
        confidence: 0,
        description: 'An error occurred during analysis. Please try again.',
        adaptations: [],
        boundingBox: null,
        camouflageRegions: []
      }),
      { 
        status: 500,
        headers: { 
          ...corsHeaders, 
          'Content-Type': 'application/json' 
        } 
      }
    );
  }
});