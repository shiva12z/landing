document.addEventListener('DOMContentLoaded', function() {
    // Add click event listeners to all Add to Cart buttons
    const addToCartButtons = document.querySelectorAll('.bg-[#0c7ff2]');
    addToCartButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Get the product details from the parent element
            const productCard = this.closest('.bg-white');
            const productName = productCard.querySelector('h3').textContent;
            const productPrice = productCard.querySelector('.text-[#0c7ff2]').textContent;
            
            // Show success message
            alert(`Added ${productName} to cart!`);
            
            // Optionally, you could add the product to a cart array here
            // For now, we'll just show the alert
        });
    });
});
