// personalized_recommendations.js
// This script dynamically loads personalized recommendations into the homepage.
// Integrate with backend API or static CSV as needed.

// Example: Fetch recommendations for the current user (mocked for now)
async function fetchRecommendations() {
  // TODO: Replace with actual API call, e.g., /api/recommendations?user_id=...
  // For now, return a static array
  return [
    {
      name: "Summer Dress",
      price: "$69.99",
      img: "https://lh3.googleusercontent.com/aida-public/AB6AXuBnWKUg8IBvbDdgS3g0Lt_ADiZuyaFVs4Z4iNbaGFEoOJCnYaF5MPnvqPmlmRxug1e-A0_P8ETIxhuXknxnAhgTruR5ILAGe9Jd1Hp4R7Lb_zPKoTAeu7Ku2XrHBEAfuDzpm4m3o5ecyB76ew5Ye_IQq0JS45qkg7gXwjVGzSSNXpfOS3leqhU3ntF5y91hGWpDP9XDBpCXexpJ7XeA0bSJM70OmUwP9giHrZJhPHCT_dao3R32wLfVj2EkQ7F8kv1hRR3GrVnn869b",
      desc: "Light and breezy summer dress, perfect for warm days and casual outings."
    },
    {
      name: "Leather Handbag",
      price: "$129.99",
      img: "https://lh3.googleusercontent.com/aida-public/AB6AXuCPq2ZffkcXgr9lgFiCqplxTnjXZ5Z1F51_hSAYreEzFbRjcFWkaA-vw7WpcSzoZyhOdi06JcxuNgNfvjenLTWTqh6Nq3CeT6g4ovuV9N-182_EcQxoAPr21jWXPJi7UI4aqyN30rTN6drW02gXN71WJenZNnVnEQbqE7qv7pHMPRG-1or8xwWcnHPrattIq6pOPWRZ4wHN0KmZYm-nPOzWqu8GeRa2HqxqfOwp2G63GwOAbp9zUT89pKmiZgvbwGrTG84WIHItcCC-",
      desc: "Premium leather handbag with modern design and spacious interior."
    },
    // ...add more as needed
  ];
}

// Render recommendations into the grid
async function renderRecommendations() {
  const recommendations = await fetchRecommendations();
  const grid = document.querySelector(
    'h2:contains("Recommended for You") + div.grid'
  );
  if (!grid) return;
  grid.innerHTML = '';
  recommendations.forEach(product => {
    const card = document.createElement('div');
    card.className = 'flex flex-col gap-3 pb-3';
    card.innerHTML = `
      <div class="w-full bg-center bg-no-repeat aspect-square bg-cover rounded-lg cursor-pointer product-card"
        data-product='${JSON.stringify(product)}'
        style='background-image: url("${product.img}");'></div>
      <div>
        <p class="text-[#0d141c] text-base font-medium leading-normal">${product.name}</p>
        <p class="text-[#49739c] text-sm font-normal leading-normal">${product.price}</p>
      </div>
    `;
    grid.appendChild(card);
  });
  // Re-attach modal logic if needed
  if (window.attachProductModalLogic) window.attachProductModalLogic();
}

// Optionally, call renderRecommendations() on page load
document.addEventListener('DOMContentLoaded', renderRecommendations);

// Export for manual use if needed
window.renderRecommendations = renderRecommendations;
