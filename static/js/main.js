/*global $, alert, jquery, console*/

$(function () {

	 'use strict'; // For Javascript Lint Error
	 
	 // Fire Nice Scroll & Set Options
 	$('html, .normal-input-text, .summary-text, textarea').niceScroll({ cursorcolor: '#C5C5D5', cursorwidth: '6px', cursorborder: 'none', background: 'transparent' });

	// Counter options ( Plus and Minus )
	$('.minus').on("click", function () {
		var $input = $(this).parent().find('input');
		var count = parseInt($input.val()) - 1;
		count = count < 1 ? 1 : count;
		$input.val(count);
		$input.change();
		return false;
	});

	$('.plus').on("click", function () {
		var $input = $(this).parent().find('input');
		$input.val(parseInt($input.val()) + 1);
		$input.change();
		return false;
	});

	//  Make the height of output panel equal to the main panel height
	$('.output-panel').height($('.main-panel').outerHeight());

 });