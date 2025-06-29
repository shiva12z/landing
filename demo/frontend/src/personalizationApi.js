export async function fetchPersonalizedContent(userData) {
  const response = await fetch('/api/personalize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(userData),
  });
  if (!response.ok) {
    throw new Error('Failed to fetch personalized content');
  }
  return await response.json();
} 