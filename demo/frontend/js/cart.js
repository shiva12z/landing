// Cart functionality for all pages
let cart = JSON.parse(localStorage.getItem('cart')) || [];

function updateCartDisplay() {
  const cartItems = document.getElementById('cart-items');
  const cartCount = document.getElementById('cart-count');
  const cartTotal = document.getElementById('cart-total');
  const cartTotalPrice = document.getElementById('cart-total-price');
  
  if (!cartItems) return; // Exit if cart elements don't exist
  
  if (cart.length === 0) {
    cartItems.innerHTML = '<p class="text-gray-500 text-center">Your cart is empty</p>';
    if (cartCount) cartCount.classList.add('hidden');
    if (cartTotal) cartTotal.classList.add('hidden');
  } else {
    if (cartCount) {
      cartCount.textContent = cart.reduce((total, item) => total + item.quantity, 0);
      cartCount.classList.remove('hidden');
    }
    if (cartTotal) cartTotal.classList.remove('hidden');
    
    let totalPrice = 0;
    cartItems.innerHTML = cart.map((item, index) => {
      const price = parseFloat(item.price.replace('$', ''));
      totalPrice += price * item.quantity;
      return `
        <div class="flex items-center gap-3 mb-3 p-2 border-b border-gray-100">
          <img src="${item.img}" alt="${item.name}" class="w-12 h-12 object-cover rounded">
          <div class="flex-1">
            <h4 class="font-medium text-sm text-[#0d141c]">${item.name}</h4>
            <p class="text-sm text-[#49739c]">${item.price}</p>
          </div>
          <div class="flex items-center gap-2">
            <button onclick="updateQuantity(${index}, -1)" class="w-6 h-6 bg-gray-200 rounded flex items-center justify-center text-sm">-</button>
            <span class="text-sm w-8 text-center">${item.quantity}</span>
            <button onclick="updateQuantity(${index}, 1)" class="w-6 h-6 bg-gray-200 rounded flex items-center justify-center text-sm">+</button>
            <button onclick="removeFromCart(${index})" class="text-red-500 text-sm ml-2">×</button>
          </div>
        </div>
      `;
    }).join('');
    
    if (cartTotalPrice) cartTotalPrice.textContent = `$${totalPrice.toFixed(2)}`;
  }
}

function addToCart(product) {
  console.log('Adding to cart:', product); // Debug log
  const existingItem = cart.find(item => item.name === product.name);
  if (existingItem) {
    existingItem.quantity += 1;
    console.log('Increased quantity for existing item:', existingItem); // Debug log
  } else {
    cart.push({
      ...product,
      quantity: 1
    });
    console.log('Added new item to cart:', product); // Debug log
  }
  localStorage.setItem('cart', JSON.stringify(cart));
  console.log('Current cart:', cart); // Debug log
  updateCartDisplay();
}

function updateQuantity(index, change) {
  cart[index].quantity += change;
  if (cart[index].quantity <= 0) {
    cart.splice(index, 1);
  }
  localStorage.setItem('cart', JSON.stringify(cart));
  updateCartDisplay();
}

function removeFromCart(index) {
  cart.splice(index, 1);
  localStorage.setItem('cart', JSON.stringify(cart));
  updateCartDisplay();
}

function initializeCart() {
  // Initialize cart display
  updateCartDisplay();
  
  // Cart dropdown functionality
  const cartBtn = document.getElementById('cart-btn');
  const cartDropdown = document.getElementById('cart-dropdown');
  
  if (cartBtn && cartDropdown) {
    let cartDropdownOpen = false;
    
    cartBtn.addEventListener('click', function(e) {
      e.preventDefault();
      cartDropdownOpen = !cartDropdownOpen;
      cartDropdown.classList.toggle('hidden');
    });
    
    document.addEventListener('click', function(e) {
      if (cartDropdownOpen && !cartBtn.contains(e.target) && !cartDropdown.contains(e.target)) {
        cartDropdown.classList.add('hidden');
        cartDropdownOpen = false;
      }
    });
  }
  
  // Checkout functionality
  const checkoutBtn = document.getElementById('checkout-btn');
  if (checkoutBtn) {
    checkoutBtn.addEventListener('click', function() {
      if (cart.length > 0) {
        alert('Thank you for your purchase! Total: ' + document.getElementById('cart-total-price').textContent);
        cart = [];
        localStorage.setItem('cart', JSON.stringify(cart));
        updateCartDisplay();
        const cartDropdown = document.getElementById('cart-dropdown');
        if (cartDropdown) cartDropdown.classList.add('hidden');
      }
    });
  }
}

// Make functions globally accessible
window.addToCart = addToCart;
window.updateQuantity = updateQuantity;
window.removeFromCart = removeFromCart;
window.updateCartDisplay = updateCartDisplay;

// Initialize cart when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeCart); 